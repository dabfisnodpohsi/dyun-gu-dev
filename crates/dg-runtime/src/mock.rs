use tracing::trace;

use dg_core::{DataFormat, DataType, Shape, Tensor, TypeCode};
use serde::Deserialize;

use crate::{
    backend::{BackendDescriptor, BackendKind, InferBackend},
    capabilities::{supports_deployment, supports_device, supports_precision},
    error::{Error, Result},
    option::{BackendConfig, BackendOptions, ModelSource, RuntimeOption},
    TensorInfo,
};

/// Mock backend options for CI and integration tests.
#[derive(Clone, Debug, PartialEq)]
pub struct MockOptions {
    pub input_infos: Vec<TensorInfo>,
    pub output_infos: Vec<TensorInfo>,
    pub echo_inputs: bool,
    pub fill_value: u8,
}

impl Default for MockOptions {
    fn default() -> Self {
        let shape = Shape::new([1, 3, 224, 224]);
        let info = TensorInfo::new(shape, DataType::F32).with_layout(DataFormat::NCHW);
        Self {
            input_infos: vec![info.clone()],
            output_infos: vec![info],
            echo_inputs: true,
            fill_value: 0,
        }
    }
}

/// Pure Rust backend used in CI.
pub struct MockBackend {
    options: MockOptions,
    input_infos: Vec<TensorInfo>,
    output_infos: Vec<TensorInfo>,
}

impl MockBackend {
    fn new() -> Self {
        Self {
            options: MockOptions::default(),
            input_infos: Vec::new(),
            output_infos: Vec::new(),
        }
    }
}

impl InferBackend for MockBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Mock
    }

    fn init(&mut self, option: &RuntimeOption) -> Result<()> {
        trace!("initializing mock backend");
        if let Some(precision) = option.precision {
            if !supports_precision(BackendKind::Mock, precision) {
                return Err(Error::UnsupportedPrecision(precision));
            }
        }
        if let Some(device) = option.device {
            if !supports_device(BackendKind::Mock, device) {
                return Err(Error::UnsupportedDevice(device));
            }
        }
        if let Some(deploy_mode) = option.deploy_mode {
            if !supports_deployment(BackendKind::Mock, deploy_mode) {
                return Err(Error::UnsupportedDeployment(deploy_mode));
            }
        }
        let BackendOptions::Mock(options) = &option.backend_options else {
            return Err(Error::InvalidOption(
                "mock backend requires Mock backend options".to_string(),
            ));
        };
        self.options = options.clone();
        self.input_infos = self.options.input_infos.clone();
        self.output_infos = if self.options.output_infos.is_empty() {
            self.input_infos.clone()
        } else {
            self.options.output_infos.clone()
        };
        Ok(())
    }

    fn reshape(&mut self, input_shapes: &[Shape]) -> Result<()> {
        if input_shapes.len() != self.input_infos.len() {
            return Err(Error::InvalidOption(
                "mock reshape shape count must match input count".to_string(),
            ));
        }
        for (info, shape) in self.input_infos.iter_mut().zip(input_shapes.iter()) {
            info.shape = shape.clone();
        }
        if self.output_infos.len() == self.input_infos.len() && self.options.echo_inputs {
            self.output_infos = self.input_infos.clone();
        }
        Ok(())
    }

    fn input_count(&self) -> usize {
        self.input_infos.len()
    }

    fn output_count(&self) -> usize {
        self.output_infos.len()
    }

    fn input_info(&self, index: usize) -> Result<&TensorInfo> {
        self.input_infos
            .get(index)
            .ok_or_else(|| Error::InvalidOption(format!("input index out of range: {index}")))
    }

    fn output_info(&self, index: usize) -> Result<&TensorInfo> {
        self.output_infos
            .get(index)
            .ok_or_else(|| Error::InvalidOption(format!("output index out of range: {index}")))
    }

    fn input_infos(&self) -> &[TensorInfo] {
        &self.input_infos
    }

    fn output_infos(&self) -> &[TensorInfo] {
        &self.output_infos
    }

    fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        if inputs.len() != self.input_infos.len() {
            return Err(Error::InvalidOption(
                "mock run input count must match configured inputs".to_string(),
            ));
        }

        let device = dg_core::CpuDevice::new();
        let mut outputs = Vec::with_capacity(self.output_infos.len());
        for (index, output_info) in self.output_infos.iter().enumerate() {
            let output = output_info.allocate(&device)?;
            if self.options.echo_inputs && index < inputs.len() {
                let bytes = inputs[index].buffer().read_bytes();
                if bytes.len() != output.buffer().len() {
                    return Err(Error::Backend(
                        "mock backend echo output size mismatch".to_string(),
                    ));
                }
                output.buffer().write_from_slice(&bytes)?;
            } else {
                let bytes = vec![self.options.fill_value; output.buffer().len()];
                output.buffer().write_from_slice(&bytes)?;
            }
            outputs.push(output);
        }
        Ok(outputs)
    }
}

fn create_mock_backend() -> Box<dyn InferBackend> {
    Box::new(MockBackend::new())
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct MockConfig {
    shape: Option<Vec<usize>>,
    output_shape: Option<Vec<usize>>,
    dtype: Option<String>,
    output_dtype: Option<String>,
    layout: Option<String>,
    echo_inputs: Option<bool>,
    fill_value: Option<u8>,
}

fn configure_mock(config: BackendConfig) -> Result<RuntimeOption> {
    let params: MockConfig = config.parse_options("mock")?;
    let shape = Shape::new(params.shape.unwrap_or_else(|| vec![1, 4]));
    let output_shape = Shape::new(params.output_shape.unwrap_or_else(|| shape.dims().to_vec()));
    let dtype = params
        .dtype
        .as_deref()
        .map(parse_dtype)
        .transpose()?
        .unwrap_or(DataType::F32);
    let output_dtype = params
        .output_dtype
        .as_deref()
        .map(parse_dtype)
        .transpose()?
        .unwrap_or(dtype);
    let layout = params
        .layout
        .as_deref()
        .map(parse_layout)
        .transpose()?
        .unwrap_or(DataFormat::NC);
    let options = BackendOptions::Mock(MockOptions {
        input_infos: vec![TensorInfo::new(shape, dtype).with_layout(layout)],
        output_infos: vec![TensorInfo::new(output_shape, output_dtype).with_layout(layout)],
        echo_inputs: params.echo_inputs.unwrap_or(true),
        fill_value: params.fill_value.unwrap_or(0),
    });
    Ok(config.into_runtime_option(BackendKind::Mock, ModelSource::Bytes(Vec::new()), options))
}

fn parse_dtype(value: &str) -> Result<DataType> {
    match value {
        "f4" => Ok(DataType::F4),
        "f8" => Ok(DataType::F8),
        "f16" => Ok(DataType::F16),
        "f32" => Ok(DataType::F32),
        "f64" => Ok(DataType::F64),
        "bf16" => Ok(DataType::BF16),
        "u8" => Ok(DataType::U8),
        "u16" => Ok(DataType::U16),
        "u32" => Ok(DataType::new(TypeCode::Uint, 32, 1)),
        "u64" => Ok(DataType::new(TypeCode::Uint, 64, 1)),
        "i4" => Ok(DataType::I4),
        "i8" => Ok(DataType::I8),
        "i16" => Ok(DataType::I16),
        "i32" => Ok(DataType::new(TypeCode::Int, 32, 1)),
        "i64" => Ok(DataType::new(TypeCode::Int, 64, 1)),
        _ => Err(Error::InvalidOption(format!(
            "unsupported mock precision: {value}"
        ))),
    }
}

fn parse_layout(value: &str) -> Result<DataFormat> {
    match value {
        "auto" => Ok(DataFormat::Auto),
        "n" => Ok(DataFormat::N),
        "nc" => Ok(DataFormat::NC),
        "nchw" => Ok(DataFormat::NCHW),
        "nhwc" => Ok(DataFormat::NHWC),
        "nc4hw" => Ok(DataFormat::NC4HW),
        "nc8hw" => Ok(DataFormat::NC8HW),
        "ncdhw" => Ok(DataFormat::NCDHW),
        "oihw" => Ok(DataFormat::OIHW),
        _ => Err(Error::InvalidOption(format!(
            "unsupported mock layout: {value}"
        ))),
    }
}

inventory::submit! {
    BackendDescriptor {
        kind: BackendKind::Mock,
        name: "mock",
        create: create_mock_backend,
        configure: configure_mock,
    }
}
