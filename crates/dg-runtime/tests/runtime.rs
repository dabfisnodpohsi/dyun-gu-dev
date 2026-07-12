use dg_core::{
    DataFormat, DataType, DeviceKind, Quantization, QuantizationScheme, Shape, Strides, Tensor,
    TensorDesc, TypeCode,
};
use dg_runtime::{
    configure_backend, validate_runtime_option, BackendConfig, BackendKind, BackendOptions,
    CoreSelection, Error, ExternalStreamHandle, InferPoll, MockOptions, ModelFormat, ModelSource,
    Runtime, RuntimeOption, TensorInfo, TensorRtOptions,
};
use serde_json::json;

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
        Error::UnsupportedBackend(BackendKind::OpenVINO)
    ));
}

#[test]
fn backend_registry_configures_mock_by_name() {
    let config = BackendConfig::new(
        None,
        json!({
            "shape": [1, 2],
            "dtype": "u8",
            "layout": "nc",
            "echo_inputs": false,
            "fill_value": 7
        }),
    )
    .with_device(DeviceKind::Cpu);

    let option = configure_backend("mock", config).expect("configure mock");
    assert_eq!(option.backend, BackendKind::Mock);
    assert_eq!(option.device, Some(DeviceKind::Cpu));
    let options = option.backend_options.as_mock().expect("mock options");
    assert_eq!(options.input_infos[0].shape.dims(), &[1, 2]);
    assert_eq!(options.input_infos[0].dtype, DataType::U8);
    assert!(!options.echo_inputs);
    assert_eq!(options.fill_value, 7);
}

#[test]
fn backend_registry_rejects_unknown_names_and_missing_models() {
    let err = configure_backend("missing", BackendConfig::new(None, serde_json::Value::Null))
        .expect_err("backend name should be rejected");
    assert_eq!(err, Error::UnsupportedBackendName("missing".to_string()));

    let err = BackendConfig::new(None, serde_json::Value::Null)
        .require_model_file("TensorRT")
        .expect_err("model path should be required");
    assert_eq!(
        err,
        Error::InvalidOption("TensorRT requires a model file path".to_string())
    );
}

#[test]
fn runtime_option_preflight_rejects_unsupported_common_capabilities() {
    let base = RuntimeOption::new(
        BackendKind::TensorRt,
        ModelSource::Bytes(Vec::new()),
        BackendOptions::TensorRt(TensorRtOptions::default()),
    );

    let err = validate_runtime_option(&base.clone().with_precision(DataType::F64))
        .expect_err("precision should be rejected");
    assert_eq!(err, Error::UnsupportedPrecision(DataType::F64));

    let err = validate_runtime_option(&base.clone().with_device(DeviceKind::Cpu))
        .expect_err("device should be rejected");
    assert_eq!(err, Error::UnsupportedDevice(DeviceKind::Cpu));

    let err = validate_runtime_option(&base.with_deploy_mode(dg_core::DeployMode::SoC))
        .expect_err("deployment should be rejected");
    assert_eq!(err, Error::UnsupportedDeployment(dg_core::DeployMode::SoC));
}

#[test]
fn runtime_option_builders_and_model_format_inference() {
    let option = RuntimeOption::new(
        BackendKind::Mock,
        ModelSource::File("model.onnx".into()),
        BackendOptions::Mock(MockOptions::default()),
    );
    assert_eq!(option.model_format, ModelFormat::Auto);
    let option = option
        .with_device_id(2)
        .with_core(CoreSelection::Single(1))
        .with_cpu_threads(4)
        .with_model_format(ModelFormat::Engine)
        .with_zero_copy(true)
        .with_dynamic_shape(true)
        .with_external_stream(ExternalStreamHandle {
            kind: dg_core::StreamKind::Cpu,
            raw: 42,
        });

    assert_eq!(option.device_id, Some(2));
    assert_eq!(option.core, CoreSelection::Single(1));
    assert_eq!(option.cpu_threads, Some(4));
    assert_eq!(option.model_format, ModelFormat::Engine);
    assert!(option.zero_copy);
    assert!(option.dynamic_shape);
    assert_eq!(
        option.external_stream,
        Some(ExternalStreamHandle {
            kind: dg_core::StreamKind::Cpu,
            raw: 42,
        })
    );

    assert_eq!(
        ModelFormat::from_source(&ModelSource::File("network.rknn".into())),
        ModelFormat::Rknn
    );
    assert_eq!(
        ModelFormat::from_source(&ModelSource::File("network.unknown".into())),
        ModelFormat::Auto
    );
    assert_eq!(
        ModelFormat::from_source(&ModelSource::Bytes(Vec::new())),
        ModelFormat::Auto
    );
}

#[test]
fn runtime_submit_poll_round_trip_and_overlap_guard() {
    let info = TensorInfo::new(Shape::new([1, 4]), DataType::U8).with_layout(DataFormat::NC);
    let option = RuntimeOption::new(
        BackendKind::Mock,
        ModelSource::Bytes(Vec::new()),
        BackendOptions::Mock(MockOptions {
            input_infos: vec![info.clone()],
            output_infos: vec![info],
            echo_inputs: true,
            fill_value: 0,
        }),
    );
    let mut runtime = Runtime::new(option).expect("construct runtime");
    assert!(matches!(runtime.poll().expect("poll"), InferPoll::Pending));

    let device = dg_core::CpuDevice::new();
    let input = Tensor::allocate(
        &device,
        TensorDesc::new(
            Shape::new([1, 4]),
            DataType::U8,
            DataFormat::NC,
            DeviceKind::Cpu,
        ),
    )
    .expect("allocate input");
    input
        .buffer()
        .write_from_slice(&[4, 3, 2, 1])
        .expect("seed input");

    runtime.submit(&[input], None).expect("submit inference");
    let overlap = runtime
        .submit(&[], None)
        .expect_err("overlapping submit should fail");
    assert!(matches!(overlap, Error::Backend(message) if message.contains("already in flight")));

    let InferPoll::Ready(outputs) = runtime.poll().expect("poll result") else {
        panic!("submission should be ready");
    };
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].buffer().read_bytes(), vec![4, 3, 2, 1]);
    assert!(matches!(runtime.poll().expect("poll"), InferPoll::Pending));
}
