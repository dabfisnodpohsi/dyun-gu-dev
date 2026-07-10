//! No-hardware tests: exercise the Sophon validation and metadata-conversion
//! logic without touching a real device. These run in the default (no-SDK) CI
//! build.

use dg_core::{DataType, DeployMode, DeviceKind, Shape, TypeCode};
use dg_runtime::{BackendKind, BackendOptions, Error, ModelSource, RuntimeOption, SophonOptions};
use dg_sophon::convert::{bm_shape_dims, byte_size, shape_from_bm, SophonDataType, BM_MAX_DIMS};
use dg_sophon::validate::{validate_deploy_mode, validate_options};

fn runtime_option(options: SophonOptions) -> RuntimeOption {
    RuntimeOption::new(
        BackendKind::Sophon,
        ModelSource::Bytes(vec![0u8; 8]),
        BackendOptions::Sophon(options),
    )
}

#[test]
fn stub_reports_backend_disabled_by_default() {
    // The `backend` feature is off in the default CI build.
    assert!(!dg_sophon::backend_enabled());
}

#[test]
fn supported_precisions_map_to_sophon_types() {
    for (dtype, expected) in [
        (DataType::F32, SophonDataType::Float32),
        (DataType::F16, SophonDataType::Float16),
        (DataType::U8, SophonDataType::Uint8),
        (DataType::I8, SophonDataType::Int8),
        (DataType::new(TypeCode::Int, 32, 1), SophonDataType::Int32),
        (DataType::new(TypeCode::Uint, 32, 1), SophonDataType::Uint32),
    ] {
        assert_eq!(SophonDataType::from_data_type(dtype).unwrap(), expected);
    }
}

#[test]
fn unsupported_precision_is_rejected_end_to_end() {
    let option = runtime_option(SophonOptions::default()).with_precision(DataType::F4);
    assert!(matches!(
        validate_options(&option, &SophonOptions::default()),
        Err(Error::UnsupportedPrecision(_))
    ));
}

#[test]
fn wrong_device_is_rejected() {
    let option = runtime_option(SophonOptions::default()).with_device(DeviceKind::RknnNpu);
    assert!(matches!(
        validate_options(&option, &SophonOptions::default()),
        Err(Error::UnsupportedDevice(DeviceKind::RknnNpu))
    ));
}

#[test]
fn deploy_mode_guard_matches_compiled_target() {
    assert!(validate_deploy_mode(DeployMode::Host, DeployMode::Host).is_ok());
    assert!(validate_deploy_mode(DeployMode::SoC, DeployMode::Host).is_err());
}

#[test]
fn shape_and_byte_size_conversions_round_trip() {
    let shape = Shape::new([1, 3, 224, 224]);
    let (num_dims, dims) = bm_shape_dims(&shape).unwrap();
    assert_eq!(num_dims, 4);
    assert_eq!(shape_from_bm(num_dims, &dims).unwrap(), shape);
    assert_eq!(
        byte_size(SophonDataType::Float32, &shape).unwrap(),
        3 * 224 * 224 * 4
    );
    assert!(bm_shape_dims(&Shape::new([1; BM_MAX_DIMS + 1])).is_err());
}
