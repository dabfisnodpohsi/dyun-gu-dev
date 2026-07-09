use dg_core::{DataType, DeployMode, DeviceKind, TypeCode};
use dg_runtime::{
    backend_capabilities, supports_deployment, supports_device, supports_precision, BackendKind,
};

#[test]
fn backend_capabilities_match_kinds() {
    for kind in [
        BackendKind::Mock,
        BackendKind::OpenVINO,
        BackendKind::Rknn,
        BackendKind::TensorRt,
        BackendKind::Sophon,
    ] {
        let caps = backend_capabilities(kind).expect("capabilities");
        assert_eq!(caps.kind, kind);
    }
}

#[test]
fn capability_matrix_reports_expected_support() {
    assert!(supports_precision(BackendKind::Mock, DataType::F4));
    assert!(supports_precision(BackendKind::OpenVINO, DataType::F16));
    assert!(supports_precision(BackendKind::Rknn, DataType::U8));
    assert!(supports_precision(BackendKind::TensorRt, DataType::F32));
    assert!(supports_precision(BackendKind::Sophon, DataType::I8));

    let unsupported = DataType::new(TypeCode::Int, 128, 1);
    assert!(!supports_precision(BackendKind::OpenVINO, unsupported));
    assert!(!supports_precision(BackendKind::Rknn, unsupported));
}

#[test]
fn capability_matrix_reports_device_and_deployment_support() {
    assert!(supports_device(BackendKind::Mock, DeviceKind::Cpu));
    assert!(supports_device(BackendKind::OpenVINO, DeviceKind::IntelGpu));
    assert!(supports_device(BackendKind::Rknn, DeviceKind::RknnNpu));
    assert!(supports_device(BackendKind::TensorRt, DeviceKind::CudaGpu));
    assert!(supports_device(BackendKind::Sophon, DeviceKind::SophonTpu));

    assert!(supports_deployment(BackendKind::Mock, DeployMode::Host));
    assert!(supports_deployment(BackendKind::Mock, DeployMode::SoC));
    assert!(supports_deployment(BackendKind::Sophon, DeployMode::Host));
    assert!(supports_deployment(BackendKind::Sophon, DeployMode::SoC));
    assert!(!supports_deployment(BackendKind::TensorRt, DeployMode::SoC));
}
