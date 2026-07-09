use std::convert::TryFrom;

use dg_core::{DataFormat, DataType, Shape, Tensor};
use dg_runtime::{
    BackendDescriptor, BackendKind, BackendOptions, Error, InferBackend, ModelSource, Result,
    RuntimeOption, TensorInfo,
};
use openvino::{
    Core, DeviceType, ElementType, InferRequest, Model, Node, PartialShape, Tensor as OvTensor,
};
use tracing::trace;

pub use dg_runtime::OpenVINOOptions;

pub fn backend_enabled() -> bool {
    true
}

pub struct OpenVINOBackend {
    option: Option<RuntimeOption>,
    core: Option<Core>,
    model: Option<Model>,
    compiled_model: Option<openvino::CompiledModel>,
    input_infos: Vec<TensorInfo>,
    output_infos: Vec<TensorInfo>,
    request: Option<InferRequest>,
}

impl OpenVINOBackend {
    pub fn new() -> Self {
        Self {
            option: None,
            core: None,
            model: None,
            compiled_model: None,
            input_infos: Vec::new(),
            output_infos: Vec::new(),
            request: None,
        }
    }

    fn map_element_type(dtype: DataType) -> Result<ElementType> {
        if dtype == DataType::F32 {
            Ok(ElementType::F32)
        } else if dtype == DataType::F16 {
            Ok(ElementType::F16)
        } else if dtype == DataType::BF16 {
            Ok(ElementType::Bf16)
        } else if dtype == DataType::U8 {
            Ok(ElementType::U8)
        } else if dtype == DataType::I8 {
            Ok(ElementType::I8)
        } else if dtype == DataType::U16 {
            Ok(ElementType::U16)
        } else if dtype == DataType::I16 {
            Ok(ElementType::I16)
        } else if dtype == DataType::new(dg_core::TypeCode::Int, 32, 1) {
            Ok(ElementType::I32)
        } else if dtype == DataType::new(dg_core::TypeCode::Uint, 32, 1) {
            Ok(ElementType::U32)
        } else if dtype == DataType::new(dg_core::TypeCode::Int, 64, 1) {
            Ok(ElementType::I64)
        } else if dtype == DataType::new(dg_core::TypeCode::Uint, 64, 1) {
            Ok(ElementType::U64)
        } else {
            Err(Error::UnsupportedPrecision(dtype))
        }
    }

    fn map_data_type(element_type: ElementType) -> Result<DataType> {
        match element_type {
            ElementType::F32 => Ok(DataType::F32),
            ElementType::F16 => Ok(DataType::F16),
            ElementType::Bf16 => Ok(DataType::BF16),
            ElementType::U8 => Ok(DataType::U8),
            ElementType::I8 => Ok(DataType::I8),
            ElementType::U16 => Ok(DataType::U16),
            ElementType::I16 => Ok(DataType::I16),
            ElementType::I32 => Ok(DataType::new(dg_core::TypeCode::Int, 32, 1)),
            ElementType::U32 => Ok(DataType::new(dg_core::TypeCode::Uint, 32, 1)),
            ElementType::I64 => Ok(DataType::new(dg_core::TypeCode::Int, 64, 1)),
            ElementType::U64 => Ok(DataType::new(dg_core::TypeCode::Uint, 64, 1)),
            other => Err(Error::Backend(format!(
                "unsupported OpenVINO element type: {other}"
            ))),
        }
    }

    fn tensor_info_from_port(port: &Node) -> Result<TensorInfo> {
        let dims = match port.get_shape() {
            Ok(shape) => shape
                .get_dimensions()
                .iter()
                .map(|dim| {
                    usize::try_from(*dim)
                        .map_err(|_| Error::Backend("negative OpenVINO dimension".to_string()))
                })
                .collect::<Result<Vec<_>>>()?,
            Err(_) => {
                let partial_shape = port
                    .get_partial_shape()
                    .map_err(|err| Error::Backend(err.to_string()))?;
                partial_shape
                    .get_dimensions()
                    .iter()
                    .map(|dimension| {
                        if dimension.is_dynamic() {
                            Ok(1usize)
                        } else {
                            usize::try_from(dimension.get_max()).map_err(|_| {
                                Error::Backend("negative OpenVINO dimension".to_string())
                            })
                        }
                    })
                    .collect::<Result<Vec<_>>>()?
            }
        };

        let mut info = TensorInfo::new(
            Shape::new(dims),
            Self::map_data_type(
                port.get_element_type()
                    .map_err(|err| Error::Backend(err.to_string()))?,
            )?,
        )
        .with_layout(DataFormat::Auto);

        if let Ok(name) = port.get_name() {
            info = info.with_name(name);
        }

        Ok(info)
    }

    fn read_model(core: &mut Core, source: &ModelSource) -> Result<Model> {
        match source {
            ModelSource::Bytes(bytes) => core
                .read_model_from_buffer(bytes, None)
                .map_err(|err| Error::BackendUnavailable(err.to_string())),
            ModelSource::File(path) => {
                let path = path.clone();
                if path.extension().and_then(|ext| ext.to_str()) == Some("xml") {
                    let weights = path.with_extension("bin");
                    if weights.exists() {
                        let model_path = path.to_str().ok_or_else(|| {
                            Error::UnsupportedModelSource("non-utf8 path".to_string())
                        })?;
                        let weights_path = weights.to_str().ok_or_else(|| {
                            Error::UnsupportedModelSource("non-utf8 path".to_string())
                        })?;
                        return core
                            .read_model_from_file(model_path, weights_path)
                            .map_err(|err| Error::BackendUnavailable(err.to_string()));
                    }
                }
                let bytes = std::fs::read(path)?;
                core.read_model_from_buffer(&bytes, None)
                    .map_err(|err| Error::BackendUnavailable(err.to_string()))
            }
        }
    }

    fn openvino_options(option: &RuntimeOption) -> Result<&dg_runtime::OpenVINOOptions> {
        let BackendOptions::OpenVINO(options) = &option.backend_options else {
            return Err(Error::InvalidOption(
                "OpenVINO backend requires OpenVINO backend options".to_string(),
            ));
        };
        Ok(options)
    }
}

impl Default for OpenVINOBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferBackend for OpenVINOBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::OpenVINO
    }

    fn init(&mut self, option: &RuntimeOption) -> Result<()> {
        trace!("initializing OpenVINO backend");
        let openvino_options = Self::openvino_options(option)?;
        let mut core = Core::new().map_err(|err| Error::BackendUnavailable(err.to_string()))?;
        let device = DeviceType::from(openvino_options.device.as_str()).to_owned();
        let model = Self::read_model(&mut core, &option.model_source)?;
        let compiled_model = core
            .compile_model(&model, device)
            .map_err(|err| Error::BackendUnavailable(err.to_string()))?;
        let input_count = compiled_model
            .get_input_size()
            .map_err(|err| Error::Backend(err.to_string()))?;
        let output_count = compiled_model
            .get_output_size()
            .map_err(|err| Error::Backend(err.to_string()))?;

        let mut input_infos = Vec::with_capacity(input_count);
        for index in 0..input_count {
            let port = compiled_model
                .get_input_by_index(index)
                .map_err(|err| Error::Backend(err.to_string()))?;
            input_infos.push(Self::tensor_info_from_port(&port)?);
        }

        let mut output_infos = Vec::with_capacity(output_count);
        for index in 0..output_count {
            let port = compiled_model
                .get_output_by_index(index)
                .map_err(|err| Error::Backend(err.to_string()))?;
            output_infos.push(Self::tensor_info_from_port(&port)?);
        }

        if let Some(requested_precision) = option.precision {
            let supported = input_infos
                .iter()
                .chain(output_infos.iter())
                .all(|info| info.dtype == requested_precision);
            if !supported {
                return Err(Error::UnsupportedPrecision(requested_precision));
            }
        }

        let mut compiled_model = compiled_model;
        let request = compiled_model
            .create_infer_request()
            .map_err(|err| Error::BackendUnavailable(err.to_string()))?;
        self.option = Some(option.clone());
        self.core = Some(core);
        self.model = Some(model);
        self.compiled_model = Some(compiled_model);
        self.input_infos = input_infos;
        self.output_infos = output_infos;
        self.request = Some(request);
        Ok(())
    }

    fn reshape(&mut self, input_shapes: &[Shape]) -> Result<()> {
        let Some(option) = &self.option else {
            return Err(Error::InvalidOption("backend not initialized".to_string()));
        };
        let Some(model) = self.model.as_ref() else {
            return Err(Error::InvalidOption("model not initialized".to_string()));
        };
        if input_shapes.len() != self.input_infos.len() {
            return Err(Error::InvalidOption(
                "reshape shape count must match model inputs".to_string(),
            ));
        }

        let mut partial_shapes = Vec::with_capacity(input_shapes.len());
        let mut input_ports = Vec::with_capacity(input_shapes.len());
        for (index, shape) in input_shapes.iter().enumerate() {
            let dims = shape
                .dims()
                .iter()
                .map(|dim| {
                    i64::try_from(*dim)
                        .map_err(|_| Error::InvalidOption("shape dimension overflow".to_string()))
                })
                .collect::<Result<Vec<_>>>()?;
            let partial_shape = PartialShape::new_static(
                i64::try_from(dims.len())
                    .map_err(|_| Error::InvalidOption("rank overflow".to_string()))?,
                &dims,
            )
            .map_err(|err| Error::Backend(err.to_string()))?;
            let port = model
                .get_input_by_index(index)
                .map_err(|err| Error::Backend(err.to_string()))?;
            input_ports.push(port);
            partial_shapes.push(partial_shape);
        }

        let pairs: Vec<(&Node, &PartialShape)> =
            input_ports.iter().zip(partial_shapes.iter()).collect();
        let Some(model) = self.model.as_mut() else {
            return Err(Error::InvalidOption("model not initialized".to_string()));
        };
        model
            .reshape_by_ports(&pairs)
            .map_err(|err| Error::Backend(err.to_string()))?;
        let device = DeviceType::from(
            option
                .backend_options
                .as_openvino()
                .map(|opts| opts.device.as_str())
                .unwrap_or("CPU"),
        )
        .to_owned();
        let compiled_model = self
            .core
            .as_mut()
            .ok_or_else(|| Error::InvalidOption("core missing".to_string()))?
            .compile_model(model, device)
            .map_err(|err| Error::BackendUnavailable(err.to_string()))?;
        let mut compiled_model = compiled_model;
        self.request = Some(
            compiled_model
                .create_infer_request()
                .map_err(|err| Error::BackendUnavailable(err.to_string()))?,
        );
        self.compiled_model = Some(compiled_model);
        let existing_infos = self.input_infos.clone();
        self.input_infos = input_shapes
            .iter()
            .enumerate()
            .map(|(index, shape)| {
                let mut info = TensorInfo::new(shape.clone(), existing_infos[index].dtype)
                    .with_layout(existing_infos[index].layout.unwrap_or(DataFormat::Auto));
                if let Some(name) = existing_infos[index].name.clone() {
                    info = info.with_name(name);
                }
                info
            })
            .collect();

        let output_count = self
            .compiled_model
            .as_ref()
            .ok_or_else(|| Error::InvalidOption("compiled model missing".to_string()))?
            .get_output_size()
            .map_err(|err| Error::Backend(err.to_string()))?;
        let mut output_infos = Vec::with_capacity(output_count);
        for index in 0..output_count {
            let port = self
                .compiled_model
                .as_ref()
                .ok_or_else(|| Error::InvalidOption("compiled model missing".to_string()))?
                .get_output_by_index(index)
                .map_err(|err| Error::Backend(err.to_string()))?;
            output_infos.push(Self::tensor_info_from_port(&port)?);
        }
        self.output_infos = output_infos;
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
                "input count must match model inputs".to_string(),
            ));
        }
        let request = self
            .request
            .as_mut()
            .ok_or_else(|| Error::InvalidOption("infer request not initialized".to_string()))?;

        for (index, input) in inputs.iter().enumerate() {
            let info = &self.input_infos[index];
            let dims = info
                .shape
                .dims()
                .iter()
                .map(|dim| {
                    i64::try_from(*dim)
                        .map_err(|_| Error::InvalidOption("shape dimension overflow".to_string()))
                })
                .collect::<Result<Vec<_>>>()?;
            let ov_shape =
                openvino::Shape::new(&dims).map_err(|err| Error::Backend(err.to_string()))?;
            let element_type = Self::map_element_type(info.dtype)?;
            let mut ov_tensor = OvTensor::new(element_type, &ov_shape)
                .map_err(|err| Error::Backend(err.to_string()))?;
            let raw = ov_tensor
                .get_raw_data_mut()
                .map_err(|err| Error::Backend(err.to_string()))?;
            let bytes = input.buffer().read_bytes();
            if bytes.len() != raw.len() {
                return Err(Error::Backend("OpenVINO input size mismatch".to_string()));
            }
            raw.copy_from_slice(&bytes);
            request
                .set_input_tensor_by_index(index, &ov_tensor)
                .map_err(|err| Error::Backend(err.to_string()))?;
        }

        request
            .infer()
            .map_err(|err| Error::Backend(err.to_string()))?;

        let device = dg_core::CpuDevice::new();
        let mut outputs = Vec::with_capacity(self.output_infos.len());
        for (index, output_info) in self.output_infos.iter().enumerate() {
            let ov_tensor = request
                .get_output_tensor_by_index(index)
                .map_err(|err| Error::Backend(err.to_string()))?;
            let output = output_info.allocate(&device)?;
            let bytes = ov_tensor
                .get_raw_data()
                .map_err(|err| Error::Backend(err.to_string()))?;
            if bytes.len() != output.buffer().len() {
                return Err(Error::Backend("OpenVINO output size mismatch".to_string()));
            }
            output.buffer().write_from_slice(bytes)?;
            outputs.push(output);
        }
        Ok(outputs)
    }
}

fn create_openvino_backend() -> Box<dyn InferBackend> {
    Box::new(OpenVINOBackend::new())
}

inventory::submit! {
    BackendDescriptor {
        kind: BackendKind::OpenVINO,
        name: "openvino",
        create: create_openvino_backend,
    }
}
