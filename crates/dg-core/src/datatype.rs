use crate::{Error, Result, Shape};
use half::{bf16, f16};

/// Storage kind for tensor elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TypeCode {
    Uint,
    Int,
    Float,
    Bfloat,
    Float8,
    Float4,
    OpaqueHandle,
}

/// A compact logical data type description.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DataType {
    pub code: TypeCode,
    pub bits: u8,
    pub lanes: u8,
}

impl DataType {
    pub const fn new(code: TypeCode, bits: u8, lanes: u8) -> Self {
        Self { code, bits, lanes }
    }

    pub const U8: Self = Self::new(TypeCode::Uint, 8, 1);
    pub const U16: Self = Self::new(TypeCode::Uint, 16, 1);
    pub const I4: Self = Self::new(TypeCode::Int, 4, 1);
    pub const I8: Self = Self::new(TypeCode::Int, 8, 1);
    pub const I16: Self = Self::new(TypeCode::Int, 16, 1);
    pub const F4: Self = Self::new(TypeCode::Float4, 4, 1);
    pub const F8: Self = Self::new(TypeCode::Float8, 8, 1);
    pub const F16: Self = Self::new(TypeCode::Float, 16, 1);
    pub const BF16: Self = Self::new(TypeCode::Bfloat, 16, 1);
    pub const F32: Self = Self::new(TypeCode::Float, 32, 1);
    pub const F64: Self = Self::new(TypeCode::Float, 64, 1);

    pub const fn size_in_bits(self) -> usize {
        (self.bits as usize) * (self.lanes as usize)
    }

    pub fn bytes_per_element_ceil(self) -> usize {
        self.size_in_bits().div_ceil(8)
    }

    pub fn storage_bytes_for_elements(self, elements: usize) -> Result<usize> {
        let total_bits = self
            .size_in_bits()
            .checked_mul(elements)
            .ok_or_else(|| Error::InvalidArgument("datatype size overflow".to_string()))?;
        Ok(total_bits.div_ceil(8))
    }

    pub fn storage_bytes_for_shape(self, shape: &Shape) -> Result<usize> {
        self.storage_bytes_for_elements(shape.element_count()?)
    }

    pub fn of<T: NativeDataType>() -> Self {
        T::data_type()
    }
}

/// Maps a native Rust value to a `DataType`.
pub trait NativeDataType {
    fn data_type() -> DataType;
}

impl NativeDataType for u8 {
    fn data_type() -> DataType {
        DataType::U8
    }
}

impl NativeDataType for i8 {
    fn data_type() -> DataType {
        DataType::I8
    }
}

impl NativeDataType for u16 {
    fn data_type() -> DataType {
        DataType::U16
    }
}

impl NativeDataType for i16 {
    fn data_type() -> DataType {
        DataType::I16
    }
}

impl NativeDataType for u32 {
    fn data_type() -> DataType {
        DataType::new(TypeCode::Uint, 32, 1)
    }
}

impl NativeDataType for i32 {
    fn data_type() -> DataType {
        DataType::new(TypeCode::Int, 32, 1)
    }
}

impl NativeDataType for u64 {
    fn data_type() -> DataType {
        DataType::new(TypeCode::Uint, 64, 1)
    }
}

impl NativeDataType for i64 {
    fn data_type() -> DataType {
        DataType::new(TypeCode::Int, 64, 1)
    }
}

impl NativeDataType for f16 {
    fn data_type() -> DataType {
        DataType::F16
    }
}

impl NativeDataType for bf16 {
    fn data_type() -> DataType {
        DataType::BF16
    }
}

impl NativeDataType for f32 {
    fn data_type() -> DataType {
        DataType::F32
    }
}

impl NativeDataType for f64 {
    fn data_type() -> DataType {
        DataType::F64
    }
}

fn pack_nibbles(values: &[u8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(2));
    for chunk in values.chunks(2) {
        let lo = chunk[0] & 0x0f;
        let hi = chunk.get(1).copied().unwrap_or(0) & 0x0f;
        packed.push(lo | (hi << 4));
    }
    packed
}

fn unpack_nibbles(bytes: &[u8], count: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(count);
    for &byte in bytes {
        values.push(byte & 0x0f);
        if values.len() == count {
            break;
        }
        values.push((byte >> 4) & 0x0f);
        if values.len() == count {
            break;
        }
    }
    values
}

/// Packs 4-bit signed integers using two's complement nibble encoding.
pub fn pack_int4(values: &[i8]) -> Result<Vec<u8>> {
    let mut raw = Vec::with_capacity(values.len());
    for &value in values {
        if !(-8..=7).contains(&value) {
            return Err(Error::InvalidArgument("int4 out of range".to_string()));
        }
        raw.push((value as i16 & 0x0f) as u8);
    }
    Ok(pack_nibbles(&raw))
}

/// Unpacks 4-bit signed integers using two's complement nibble encoding.
pub fn unpack_int4(bytes: &[u8], count: usize) -> Result<Vec<i8>> {
    let raw = unpack_nibbles(bytes, count);
    Ok(raw
        .into_iter()
        .map(|nibble| {
            if nibble & 0x08 != 0 {
                (nibble as i8) - 16
            } else {
                nibble as i8
            }
        })
        .collect())
}

/// Packs 4-bit floating payloads as raw nibbles.
pub fn pack_float4(values: &[u8]) -> Result<Vec<u8>> {
    if values.iter().any(|&value| value > 0x0f) {
        return Err(Error::InvalidArgument("float4 out of range".to_string()));
    }
    Ok(pack_nibbles(values))
}

/// Unpacks 4-bit floating payloads as raw nibbles.
pub fn unpack_float4(bytes: &[u8], count: usize) -> Result<Vec<u8>> {
    Ok(unpack_nibbles(bytes, count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int4_round_trip() {
        let values = [-8, -3, 0, 1, 7];
        let packed = pack_int4(&values).expect("pack int4");
        let unpacked = unpack_int4(&packed, values.len()).expect("unpack int4");
        assert_eq!(unpacked, values);
    }

    #[test]
    fn float4_round_trip() {
        let values = [0, 1, 2, 3, 14, 15];
        let packed = pack_float4(&values).expect("pack float4");
        let unpacked = unpack_float4(&packed, values.len()).expect("unpack float4");
        assert_eq!(unpacked, values);
    }
}
