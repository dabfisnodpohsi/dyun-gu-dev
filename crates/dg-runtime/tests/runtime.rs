use dg_core::{
    DataFormat, DataType, DeviceKind, Quantization, QuantizationScheme, Shape, Strides, Tensor,
    TensorDesc, TypeCode,
};
use dg_runtime::{
    BackendKind, BackendOptions, MockOptions, ModelSource, Runtime, RuntimeOption, TensorInfo,
};

#[test]
fn mock_backend_registry_and_run_identity() {
    let input_info = TensorInfo::new(Shape::new([1, 4]), DataType::U8).with_layout(DataFormat::NC);
    let output_info = input_info.clone();
    let option = RuntimeOption::new(
        BackendKind::Mock,
        ModelSource::Bytes(Vec::new()),
        BackendOptions::Mock(MockOptions {
            input_infos: vec![input_info.clone()],
            output_infos: vec![output_info.clone()],
            echo_inputs: true,
            fill_value: 0,
        }),
    );

    let mut runtime = Runtime::new(option).expect("construct runtime");
    assert_eq!(runtime.backend_kind(), BackendKind::Mock);
    assert_eq!(runtime.input_infos(), &[input_info]);
    assert_eq!(runtime.output_infos(), &[output_info]);

    let device = dg_core::CpuDevice::new();
    let desc = TensorDesc::new(
        Shape::new([1, 4]),
        DataType::U8,
        DataFormat::NC,
        DeviceKind::Cpu,
    );
    let input = Tensor::allocate(&device, desc).expect("allocate input");
    input
        .buffer()
        .write_from_slice(&[1, 2, 3, 4])
        .expect("seed input");

    let outputs = runtime.run(&[input]).expect("run backend");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].buffer().read_bytes(), vec![1, 2, 3, 4]);
}

#[test]
fn mock_backend_rejects_unsupported_precision() {
    let option = RuntimeOption::new(
        BackendKind::Mock,
        ModelSource::Bytes(Vec::new()),
        BackendOptions::Mock(MockOptions::default()),
    )
    .with_precision(DataType::new(TypeCode::OpaqueHandle, 8, 1));

    let err = Runtime::new(option)
        .err()
        .expect("precision should be rejected");
    assert!(matches!(err, dg_runtime::Error::UnsupportedPrecision(_)));
}

#[test]
fn tensor_info_carries_quantization_and_stride_metadata() {
    let quant = Quantization {
        scheme: QuantizationScheme::AffineAsymmetric,
        scale: vec![0.5],
        zero_point: vec![-3],
        axis: None,
    };
    let strides = Strides::new([3 * 224 * 256, 224 * 256, 256, 1]);
    let info = TensorInfo::new(Shape::new([1, 3, 224, 224]), DataType::I8)
        .with_layout(DataFormat::NCHW)
        .with_quantization(quant.clone())
        .with_strides(strides.clone())
        .with_size_with_stride(3 * 224 * 256);

    assert!(info.is_quantized());
    assert_eq!(info.size_with_stride, Some(3 * 224 * 256));

    let desc = info.tensor_desc(DeviceKind::Cpu);
    assert_eq!(desc.quantization(), &quant);
    assert_eq!(desc.strides(), Some(&strides));
}

#[test]
fn unknown_backend_is_rejected() {
    let option = RuntimeOption::new(
        BackendKind::OpenVINO,
        ModelSource::Bytes(Vec::new()),
        BackendOptions::OpenVINO(Default::default()),
    );

    let err = Runtime::new(option)
        .err()
        .expect("backend should be missing");
    assert!(matches!(
        err,
        dg_runtime::Error::UnsupportedBackend(BackendKind::OpenVINO)
    ));
}
