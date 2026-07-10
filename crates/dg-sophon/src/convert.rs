//! Pure conversions between `dg-core` tensor metadata and Sophon BMRuntime
//! types.
//!
//! These helpers are intentionally free of any FFI / `sys` dependency so they
//! compile and are unit-tested on machines without the Sophon SDK (the default
//! no-hardware build).

use dg_core::{DataType, Shape, TypeCode};
use dg_runtime::{Error, Result};

/// Maximum number of shape dimensions BMRuntime's `bm_shape_t` can carry.
pub const BM_MAX_DIMS: usize = 8;

/// Logical mirror of BMRuntime's `bm_data_type_t` enumeration.
///
/// The discriminants match the vendor ABI so [`SophonDataType::code`] can be
/// used to build FFI values without pulling `bindgen` output into pure logic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SophonDataType {
    Float32,
    Float16,
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Bfloat16,
}

impl SophonDataType {
    /// Vendor ABI code (`bm_data_type_t`) as the unsigned FFI representation.
    pub const fn bm_code(self) -> u32 {
        self.code() as u32
    }

    /// Vendor ABI code (`bm_data_type_t`).
    pub const fn code(self) -> i32 {
        match self {
            Self::Float32 => 0,
            Self::Float16 => 1,
            Self::Int8 => 2,
            Self::Uint8 => 3,
            Self::Int16 => 4,
            Self::Uint16 => 5,
            Self::Int32 => 6,
            Self::Uint32 => 7,
            Self::Bfloat16 => 8,
        }
    }

    /// Rebuilds a data type from a vendor ABI code, rejecting unknown values.
    pub fn from_code(code: i32) -> Result<Self> {
        match code {
            0 => Ok(Self::Float32),
            1 => Ok(Self::Float16),
            2 => Ok(Self::Int8),
            3 => Ok(Self::Uint8),
            4 => Ok(Self::Int16),
            5 => Ok(Self::Uint16),
            6 => Ok(Self::Int32),
            7 => Ok(Self::Uint32),
            8 => Ok(Self::Bfloat16),
            other => Err(Error::Backend(format!(
                "unsupported Sophon bm_data_type code: {other}"
            ))),
        }
    }

    /// Byte width of a single element.
    pub const fn bytes_per_element(self) -> usize {
        match self {
            Self::Int8 | Self::Uint8 => 1,
            Self::Float16 | Self::Bfloat16 | Self::Int16 | Self::Uint16 => 2,
            Self::Float32 | Self::Int32 | Self::Uint32 => 4,
        }
    }

    /// Maps a `dg-core` [`DataType`] onto the matching Sophon element type.
    ///
    /// Sub-byte, 64-bit, and opaque handle types have no BMRuntime equivalent
    /// and are rejected with an explicit error rather than silently coerced.
    pub fn from_data_type(dtype: DataType) -> Result<Self> {
        if dtype.lanes != 1 {
            return Err(Error::UnsupportedPrecision(dtype));
        }
        let mapped = match (dtype.code, dtype.bits) {
            (TypeCode::Float, 32) => Self::Float32,
            (TypeCode::Float, 16) => Self::Float16,
            (TypeCode::Bfloat, 16) => Self::Bfloat16,
            (TypeCode::Int, 8) => Self::Int8,
            (TypeCode::Uint, 8) => Self::Uint8,
            (TypeCode::Int, 16) => Self::Int16,
            (TypeCode::Uint, 16) => Self::Uint16,
            (TypeCode::Int, 32) => Self::Int32,
            (TypeCode::Uint, 32) => Self::Uint32,
            _ => return Err(Error::UnsupportedPrecision(dtype)),
        };
        Ok(mapped)
    }

    /// Maps back onto a `dg-core` [`DataType`].
    pub const fn to_data_type(self) -> DataType {
        match self {
            Self::Float32 => DataType::F32,
            Self::Float16 => DataType::F16,
            Self::Bfloat16 => DataType::BF16,
            Self::Int8 => DataType::I8,
            Self::Uint8 => DataType::U8,
            Self::Int16 => DataType::I16,
            Self::Uint16 => DataType::U16,
            Self::Int32 => DataType::new(TypeCode::Int, 32, 1),
            Self::Uint32 => DataType::new(TypeCode::Uint, 32, 1),
        }
    }
}

/// Converts a logical [`Shape`] into BMRuntime `bm_shape_t` dimensions.
///
/// Returns `(num_dims, dims)` where `dims` is zero-padded to [`BM_MAX_DIMS`].
/// Fails when the rank exceeds BMRuntime's limit or a dimension overflows the
/// signed 32-bit dimension field.
pub fn bm_shape_dims(shape: &Shape) -> Result<(i32, [i32; BM_MAX_DIMS])> {
    if shape.rank() > BM_MAX_DIMS {
        return Err(Error::InvalidOption(format!(
            "Sophon shape rank {} exceeds bm_shape_t limit {BM_MAX_DIMS}",
            shape.rank()
        )));
    }
    let mut dims = [0i32; BM_MAX_DIMS];
    for (slot, &dim) in dims.iter_mut().zip(shape.dims()) {
        *slot = i32::try_from(dim).map_err(|_| {
            Error::InvalidOption("Sophon shape dimension overflows i32".to_string())
        })?;
    }
    let num_dims = i32::try_from(shape.rank())
        .map_err(|_| Error::InvalidOption("Sophon shape rank overflows i32".to_string()))?;
    Ok((num_dims, dims))
}

/// Builds a [`Shape`] from BMRuntime `bm_shape_t` fields.
pub fn shape_from_bm(num_dims: i32, dims: &[i32]) -> Result<Shape> {
    let rank = usize::try_from(num_dims)
        .map_err(|_| Error::Backend("Sophon reported a negative dimension count".to_string()))?;
    if rank > dims.len() {
        return Err(Error::Backend(format!(
            "Sophon reported {rank} dims but only {} are available",
            dims.len()
        )));
    }
    let mut extents = Vec::with_capacity(rank);
    for &dim in dims.iter().take(rank) {
        let extent = usize::try_from(dim)
            .map_err(|_| Error::Backend("Sophon reported a negative dimension".to_string()))?;
        extents.push(extent);
    }
    Ok(Shape::new(extents))
}

/// Byte size of a densely packed tensor with `shape` and `dtype`.
pub fn byte_size(dtype: SophonDataType, shape: &Shape) -> Result<usize> {
    let elements = shape.element_count()?;
    elements
        .checked_mul(dtype.bytes_per_element())
        .ok_or_else(|| Error::InvalidOption("Sophon tensor byte size overflow".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_type_round_trips() {
        for dtype in [
            SophonDataType::Float32,
            SophonDataType::Float16,
            SophonDataType::Bfloat16,
            SophonDataType::Int8,
            SophonDataType::Uint8,
            SophonDataType::Int16,
            SophonDataType::Uint16,
            SophonDataType::Int32,
            SophonDataType::Uint32,
        ] {
            let core = dtype.to_data_type();
            assert_eq!(SophonDataType::from_data_type(core).unwrap(), dtype);
            assert_eq!(SophonDataType::from_code(dtype.code()).unwrap(), dtype);
        }
    }

    #[test]
    fn rejects_unsupported_data_types() {
        assert!(matches!(
            SophonDataType::from_data_type(DataType::I4),
            Err(Error::UnsupportedPrecision(_))
        ));
        assert!(matches!(
            SophonDataType::from_data_type(DataType::F64),
            Err(Error::UnsupportedPrecision(_))
        ));
        assert!(matches!(
            SophonDataType::from_data_type(DataType::new(TypeCode::Float, 32, 4)),
            Err(Error::UnsupportedPrecision(_))
        ));
    }

    #[test]
    fn rejects_unknown_codes() {
        assert!(SophonDataType::from_code(-1).is_err());
        assert!(SophonDataType::from_code(99).is_err());
    }

    #[test]
    fn byte_sizes_match_dtype_width() {
        let shape = Shape::new([1, 3, 4]);
        assert_eq!(byte_size(SophonDataType::Uint8, &shape).unwrap(), 12);
        assert_eq!(byte_size(SophonDataType::Float16, &shape).unwrap(), 24);
        assert_eq!(byte_size(SophonDataType::Float32, &shape).unwrap(), 48);
    }

    #[test]
    fn shape_conversion_round_trips() {
        let shape = Shape::new([1, 3, 224, 224]);
        let (num_dims, dims) = bm_shape_dims(&shape).unwrap();
        assert_eq!(num_dims, 4);
        assert_eq!(&dims[..4], &[1, 3, 224, 224]);
        assert_eq!(shape_from_bm(num_dims, &dims).unwrap(), shape);
    }

    #[test]
    fn shape_rank_limit_is_enforced() {
        let shape = Shape::new([1; BM_MAX_DIMS + 1]);
        assert!(bm_shape_dims(&shape).is_err());
    }

    #[test]
    fn shape_from_bm_rejects_negative() {
        assert!(shape_from_bm(-1, &[0; BM_MAX_DIMS]).is_err());
        assert!(shape_from_bm(1, &[-4, 0, 0, 0, 0, 0, 0, 0]).is_err());
    }
}
