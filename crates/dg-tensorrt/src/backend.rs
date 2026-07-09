use std::fs;
use std::os::raw::c_void;

use dg_core::{DataFormat, DataType, Shape, Tensor};
use dg_runtime::{
    supports_deployment, supports_device, supports_precision, BackendDescriptor, BackendKind,
    BackendOptions, Error, InferBackend, Result, RuntimeOption, TensorInfo, TensorRtOptions,
};
use tracing::trace;

#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code
)]
mod sys {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub const fn backend_enabled() -> bool {
    true
}

struct ModelBuffer {
    data: Vec<u8>,
}

impl ModelBuffer {
    fn new(data: Vec<u8>) -> Result<Self> {
        let _: u32 = data
            .len()
            .try_into()
            .map_err(|_| Error::InvalidOption("TensorRT model is too large".to_string()))?;
        Ok(Self { data })
    }

    fn as_ptr(&self) -> *const c_void {
        self.data.as_ptr() as *const c_void
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

pub struct TensorRtBackend {
    runtime: Option<*mut sys::trt_runtime_handle>,
    engine: Option<*mut sys::trt_engine_handle>,
    context: Option<*mut sys::trt_context_handle>,
    options: TensorRtOptions,
    input_infos: Vec<TensorInfo>,
    output_infos: Vec<TensorInfo>,
}

impl TensorRtBackend {
    fn new() -> Self {
        Self {
            runtime: None,
            engine: None,
            context: None,
            options: TensorRtOptions::default(),
            input_infos: Vec::new(),
            output_infos: Vec::new(),
        }
    }

    fn load_model(&self, source: &dg_runtime::ModelSource) -> Result<ModelBuffer> {
        match source {
            dg_runtime::ModelSource::File(path) => {
                let data = fs::read(path).map_err(|err| {
                    Error::BackendUnavailable(format!(
                        "failed to read TensorRT engine file {}: {err}",
                        path.display()
                    ))
                })?;
                ModelBuffer::new(data)
            }
            dg_runtime::ModelSource::Bytes(bytes) => ModelBuffer::new(bytes.clone()),
        }
    }

    fn validate(option: &RuntimeOption) -> Result<()> {
        if let Some(precision) = option.precision
            && !supports_precision(BackendKind::TensorRt, precision)
        {
            return Err(Error::UnsupportedPrecision(precision));
        }
        if let Some(device) = option.device
            && !supports_device(BackendKind::TensorRt, device)
        {
            return Err(Error::UnsupportedDevice(device));
        }
        if let Some(deploy_mode) = option.deploy_mode
            && !supports_deployment(BackendKind::TensorRt, deploy_mode)
        {
            return Err(Error::UnsupportedDeployment(deploy_mode));
        }
        Ok(())
    }
}

impl InferBackend for TensorRtBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::TensorRt
    }

    fn init(&mut self, option: &RuntimeOption) -> Result<()> {
        trace!("initializing TensorRT backend");
        Self::validate(option)?;
        let BackendOptions::TensorRt(tensorrt) = &option.backend_options else {
            return Err(Error::InvalidOption(
                "TensorRT backend requires TensorRT backend options".to_string(),
            ));
        };
        self.options = tensorrt.clone();

        let model = self.load_model(&option.model_source)?;
        let runtime = unsafe { sys::trt_runtime_create() };
        if runtime.is_null() {
            return Err(Error::BackendUnavailable(
                "trt_runtime_create returned null".to_string(),
            ));
        }
        self.runtime = Some(runtime);
        let engine =
            unsafe { sys::trt_runtime_deserialize_engine(runtime, model.as_ptr(), model.len()) };
        if engine.is_null() {
            return Err(Error::BackendUnavailable(
                "trt_runtime_deserialize_engine failed".to_string(),
            ));
        }
        self.engine = Some(engine);
        let context = unsafe { sys::trt_engine_create_context(engine) };
        if context.is_null() {
            return Err(Error::BackendUnavailable(
                "trt_engine_create_context failed".to_string(),
            ));
        }

        self.context = Some(context);
        self.input_infos =
            vec![TensorInfo::new(Shape::new([1]), DataType::F32).with_layout(DataFormat::Auto)];
        self.output_infos = self.input_infos.clone();
        Ok(())
    }

    fn reshape(&mut self, input_shapes: &[Shape]) -> Result<()> {
        if input_shapes.is_empty() {
            return Err(Error::InvalidOption(
                "TensorRT reshape requires at least one input shape".to_string(),
            ));
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
        if inputs.is_empty() {
            return Err(Error::InvalidOption(
                "TensorRT backend requires at least one input tensor".to_string(),
            ));
        }
        Err(Error::BackendUnavailable(
            "TensorRT FFI execution path requires SDK validation".to_string(),
        ))
    }
}

impl Drop for TensorRtBackend {
    fn drop(&mut self) {
        if let Some(context) = self.context.take() {
            unsafe { sys::trt_context_destroy(context) };
        }
        if let Some(engine) = self.engine.take() {
            unsafe { sys::trt_engine_destroy(engine) };
        }
        if let Some(runtime) = self.runtime.take() {
            unsafe { sys::trt_runtime_destroy(runtime) };
        }
    }
}

fn create_tensorrt_backend() -> Box<dyn InferBackend> {
    Box::new(TensorRtBackend::new())
}

inventory::submit! {
    BackendDescriptor {
        kind: BackendKind::TensorRt,
        name: "tensorrt",
        create: create_tensorrt_backend,
    }
}
