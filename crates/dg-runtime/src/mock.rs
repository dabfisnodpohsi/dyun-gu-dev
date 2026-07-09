use tracing::trace;

use dg_core::{DataFormat, DataType, Shape, Tensor};

use crate::{
    backend::{BackendDescriptor, BackendKind, InferBackend},
    error::{Error, Result},
    option::{BackendOptions, RuntimeOption},
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

    fn validate_precision(option: &RuntimeOption) -> Result<()> {
        let Some(precision) = option.precision else {
            return Ok(());
        };

        let supported = precision == DataType::F32
            || precision == DataType::F16
            || precision == DataType::BF16
            || precision == DataType::U8
            || precision == DataType::I8
            || precision == DataType::I16
            || precision == DataType::U16
            || precision == DataType::new(dg_core::TypeCode::Int, 32, 1)
            || precision == DataType::new(dg_core::TypeCode::Uint, 32, 1);
        if supported {
            Ok(())
        } else {
            Err(Error::UnsupportedPrecision(precision))
        }
    }
}

impl InferBackend for MockBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Mock
    }

    fn init(&mut self, option: &RuntimeOption) -> Result<()> {
        trace!("initializing mock backend");
        Self::validate_precision(option)?;
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

inventory::submit! {
    BackendDescriptor {
        kind: BackendKind::Mock,
        name: "mock",
        create: create_mock_backend,
    }
}
