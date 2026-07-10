//! Pure conversions between TensorRT shim values and `dg-core` types plus
//! runtime option validation. No FFI here so everything is testable without
//! CUDA hardware or the TensorRT SDK.

use dg_core::{DataType, Shape, TypeCode};
use dg_runtime::{
    supports_deployment, supports_device, supports_precision, BackendKind, Error, Result,
    RuntimeOption,
};

/// Maximum tensor rank supported by TensorRT (`nvinfer1::Dims::MAX_DIMS`).
pub(crate) const TRT_MAX_DIMS: usize = 8;

pub(crate) const TRT_DTYPE_FLOAT: i32 = 0;
pub(crate) const TRT_DTYPE_HALF: i32 = 1;
pub(crate) const TRT_DTYPE_INT8: i32 = 2;
pub(crate) const TRT_DTYPE_INT32: i32 = 3;
pub(crate) const TRT_DTYPE_BOOL: i32 = 4;
pub(crate) const TRT_DTYPE_UINT8: i32 = 5;
pub(crate) const TRT_DTYPE_FP8: i32 = 6;
pub(crate) const TRT_DTYPE_BF16: i32 = 7;
pub(crate) const TRT_DTYPE_INT64: i32 = 8;
pub(crate) const TRT_DTYPE_INT4: i32 = 9;

/// Maps a `nvinfer1::DataType` enum value onto the unified `dg-core` type.
pub(crate) fn trt_dtype_to_datatype(value: i32) -> Result<DataType> {
    match value {
        TRT_DTYPE_FLOAT => Ok(DataType::F32),
        TRT_DTYPE_HALF => Ok(DataType::F16),
        TRT_DTYPE_INT8 => Ok(DataType::I8),
        TRT_DTYPE_INT32 => Ok(DataType::new(TypeCode::Int, 32, 1)),
        TRT_DTYPE_UINT8 => Ok(DataType::U8),
        TRT_DTYPE_FP8 => Ok(DataType::F8),
        TRT_DTYPE_BF16 => Ok(DataType::BF16),
        TRT_DTYPE_INT64 => Ok(DataType::new(TypeCode::Int, 64, 1)),
        TRT_DTYPE_INT4 => Ok(DataType::I4),
        TRT_DTYPE_BOOL => Err(Error::Backend(
            "TensorRT bool tensors are not supported".to_string(),
        )),
        other => Err(Error::Backend(format!(
            "unknown TensorRT data type: {other}"
        ))),
    }
}

/// Converts resolved (non-dynamic) TensorRT dims into a `Shape`.
pub(crate) fn dims_to_shape(dims: &[i64], tensor: &str) -> Result<Shape> {
    let mut resolved = Vec::with_capacity(dims.len());
    for dim in dims {
        let dim = usize::try_from(*dim).map_err(|_| {
            Error::Backend(format!(
                "tensor {tensor} has unresolved dynamic dimension {dim}"
            ))
        })?;
        resolved.push(dim);
    }
    Ok(Shape::new(resolved))
}

/// Converts engine-level dims into a `Shape`, substituting `1` for dynamic
/// (`-1`) dimensions. Returns whether any dimension was dynamic.
pub(crate) fn engine_dims_to_shape(dims: &[i64], tensor: &str) -> Result<(Shape, bool)> {
    let mut resolved = Vec::with_capacity(dims.len());
    let mut dynamic = false;
    for dim in dims {
        if *dim == -1 {
            dynamic = true;
            resolved.push(1usize);
        } else {
            let dim = usize::try_from(*dim).map_err(|_| {
                Error::Backend(format!("tensor {tensor} has invalid dimension {dim}"))
            })?;
            resolved.push(dim);
        }
    }
    Ok((Shape::new(resolved), dynamic))
}

/// Converts a `Shape` into TensorRT dims, enforcing the rank limit.
pub(crate) fn shape_to_dims(shape: &Shape, tensor: &str) -> Result<Vec<i64>> {
    let dims = shape.dims();
    if dims.len() > TRT_MAX_DIMS {
        return Err(Error::InvalidOption(format!(
            "tensor {tensor} rank {} exceeds TensorRT maximum {TRT_MAX_DIMS}",
            dims.len()
        )));
    }
    dims.iter()
        .map(|dim| {
            i64::try_from(*dim).map_err(|_| {
                Error::InvalidOption(format!("tensor {tensor} dimension {dim} overflows i64"))
            })
        })
        .collect()
}

/// Checks a concrete shape against engine dims where `-1` marks a dynamic
/// dimension that any extent may satisfy.
pub(crate) fn shape_matches_engine_dims(shape: &Shape, engine_dims: &[i64]) -> bool {
    let dims = shape.dims();
    if dims.len() != engine_dims.len() {
        return false;
    }
    dims.iter().zip(engine_dims).all(|(dim, engine_dim)| {
        *engine_dim == -1 || i64::try_from(*dim).is_ok_and(|dim| dim == *engine_dim)
    })
}

/// Validates precision/device/deployment requests against the static TensorRT
/// capability matrix before touching the SDK.
pub(crate) fn validate_runtime_option(option: &RuntimeOption) -> Result<()> {
    if let Some(precision) = option.precision {
        if !supports_precision(BackendKind::TensorRt, precision) {
            return Err(Error::UnsupportedPrecision(precision));
        }
    }
    if let Some(device) = option.device {
        if !supports_device(BackendKind::TensorRt, device) {
            return Err(Error::UnsupportedDevice(device));
        }
    }
    if let Some(deploy_mode) = option.deploy_mode {
        if !supports_deployment(BackendKind::TensorRt, deploy_mode) {
            return Err(Error::UnsupportedDeployment(deploy_mode));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use dg_core::{DeployMode, DeviceKind};
    use dg_runtime::{BackendOptions, ModelSource, TensorRtOptions};

    use super::*;

    fn tensorrt_option() -> RuntimeOption {
        RuntimeOption::new(
            BackendKind::TensorRt,
            ModelSource::Bytes(vec![0u8; 4]),
            BackendOptions::TensorRt(TensorRtOptions::default()),
        )
    }

    #[test]
    fn dtype_roundtrip_supported() {
        assert_eq!(trt_dtype_to_datatype(TRT_DTYPE_FLOAT), Ok(DataType::F32));
        assert_eq!(trt_dtype_to_datatype(TRT_DTYPE_HALF), Ok(DataType::F16));
        assert_eq!(trt_dtype_to_datatype(TRT_DTYPE_INT8), Ok(DataType::I8));
        assert_eq!(trt_dtype_to_datatype(TRT_DTYPE_UINT8), Ok(DataType::U8));
        assert_eq!(trt_dtype_to_datatype(TRT_DTYPE_BF16), Ok(DataType::BF16));
        assert_eq!(trt_dtype_to_datatype(TRT_DTYPE_FP8), Ok(DataType::F8));
        assert_eq!(trt_dtype_to_datatype(TRT_DTYPE_INT4), Ok(DataType::I4));
        assert_eq!(
            trt_dtype_to_datatype(TRT_DTYPE_INT32),
            Ok(DataType::new(TypeCode::Int, 32, 1))
        );
        assert_eq!(
            trt_dtype_to_datatype(TRT_DTYPE_INT64),
            Ok(DataType::new(TypeCode::Int, 64, 1))
        );
    }

    #[test]
    fn dtype_rejects_bool_and_unknown() {
        assert!(trt_dtype_to_datatype(TRT_DTYPE_BOOL).is_err());
        assert!(trt_dtype_to_datatype(42).is_err());
    }

    #[test]
    fn dims_to_shape_rejects_dynamic() {
        assert_eq!(
            dims_to_shape(&[1, 3, 224, 224], "input").unwrap().dims(),
            &[1, 3, 224, 224]
        );
        assert!(dims_to_shape(&[1, -1], "input").is_err());
    }

    #[test]
    fn engine_dims_substitute_dynamic() {
        let (shape, dynamic) = engine_dims_to_shape(&[-1, 3, 224, 224], "input").unwrap();
        assert!(dynamic);
        assert_eq!(shape.dims(), &[1, 3, 224, 224]);

        let (shape, dynamic) = engine_dims_to_shape(&[1, 3], "input").unwrap();
        assert!(!dynamic);
        assert_eq!(shape.dims(), &[1, 3]);

        assert!(engine_dims_to_shape(&[-2, 3], "input").is_err());
    }

    #[test]
    fn shape_to_dims_enforces_rank_limit() {
        let shape = Shape::new([1usize, 2, 3]);
        assert_eq!(shape_to_dims(&shape, "input").unwrap(), vec![1i64, 2, 3]);

        let too_deep = Shape::new([1usize; TRT_MAX_DIMS + 1]);
        assert!(shape_to_dims(&too_deep, "input").is_err());
    }

    #[test]
    fn shape_matching_honours_dynamic_dims() {
        let shape = Shape::new([2usize, 3, 8]);
        assert!(shape_matches_engine_dims(&shape, &[2, 3, 8]));
        assert!(shape_matches_engine_dims(&shape, &[-1, 3, 8]));
        assert!(!shape_matches_engine_dims(&shape, &[2, 3, 9]));
        assert!(!shape_matches_engine_dims(&shape, &[2, 3]));
    }

    #[test]
    fn validation_accepts_supported_matrix() {
        let option = tensorrt_option()
            .with_precision(DataType::F16)
            .with_device(DeviceKind::CudaGpu)
            .with_deploy_mode(DeployMode::Host);
        assert!(validate_runtime_option(&option).is_ok());
    }

    #[test]
    fn validation_rejects_unsupported_requests() {
        assert_eq!(
            validate_runtime_option(&tensorrt_option().with_precision(DataType::F64)),
            Err(Error::UnsupportedPrecision(DataType::F64))
        );
        assert_eq!(
            validate_runtime_option(&tensorrt_option().with_device(DeviceKind::RknnNpu)),
            Err(Error::UnsupportedDevice(DeviceKind::RknnNpu))
        );
        assert_eq!(
            validate_runtime_option(&tensorrt_option().with_deploy_mode(DeployMode::SoC)),
            Err(Error::UnsupportedDeployment(DeployMode::SoC))
        );
    }
}
