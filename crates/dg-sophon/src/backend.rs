use std::fs;
use std::os::raw::c_void;

use dg_core::{DataFormat, DataType, Shape, Tensor};
use dg_runtime::{
    supports_deployment, supports_device, supports_precision, BackendDescriptor, BackendKind,
    BackendOptions, Error, InferBackend, Result, RuntimeOption, SophonOptions, TensorInfo,
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
            .map_err(|_| Error::InvalidOption("Sophon model is too large".to_string()))?;
        Ok(Self { data })
    }

    fn as_ptr(&self) -> *const c_void {
        self.data.as_ptr() as *const c_void
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

pub struct SophonBackend {
    handle: Option<sys::bm_handle_t>,
    runtime: Option<*mut c_void>,
    options: SophonOptions,
    input_infos: Vec<TensorInfo>,
    output_infos: Vec<TensorInfo>,
}

impl SophonBackend {
    fn new() -> Self {
        Self {
            handle: None,
            runtime: None,
            options: SophonOptions::default(),
            input_infos: Vec::new(),
            output_infos: Vec::new(),
        }
    }

    fn load_model(&self, source: &dg_runtime::ModelSource) -> Result<ModelBuffer> {
        match source {
            dg_runtime::ModelSource::File(path) => {
                let data = fs::read(path).map_err(|err| {
                    Error::BackendUnavailable(format!(
                        "failed to read Sophon model file {}: {err}",
                        path.display()
                    ))
                })?;
                ModelBuffer::new(data)
            }
            dg_runtime::ModelSource::Bytes(bytes) => ModelBuffer::new(bytes.clone()),
        }
    }

    fn validate(option: &RuntimeOption, sophon: &SophonOptions) -> Result<()> {
        if let Some(precision) = option.precision
            && !supports_precision(BackendKind::Sophon, precision)
        {
            return Err(Error::UnsupportedPrecision(precision));
        }
        if let Some(device) = option.device
            && !supports_device(BackendKind::Sophon, device)
        {
            return Err(Error::UnsupportedDevice(device));
        }
        if let Some(deploy_mode) = option.deploy_mode
            && !supports_deployment(BackendKind::Sophon, deploy_mode)
        {
            return Err(Error::UnsupportedDeployment(deploy_mode));
        }
        if !supports_deployment(BackendKind::Sophon, sophon.deploy_mode) {
            return Err(Error::UnsupportedDeployment(sophon.deploy_mode));
        }
        Ok(())
    }
}

impl InferBackend for SophonBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Sophon
    }

    fn init(&mut self, option: &RuntimeOption) -> Result<()> {
        trace!("initializing Sophon backend");
        let BackendOptions::Sophon(sophon) = &option.backend_options else {
            return Err(Error::InvalidOption(
                "Sophon backend requires Sophon backend options".to_string(),
            ));
        };
        Self::validate(option, sophon)?;

        let model = self.load_model(&option.model_source)?;
        let device_id = sophon.device_id.unwrap_or(0);
        let mut handle: sys::bm_handle_t = std::ptr::null_mut();
        let status = unsafe { sys::bm_dev_request(&mut handle, device_id as i32) };
        if status != sys::bm_status_t::BM_SUCCESS {
            return Err(Error::Backend(format!("bm_dev_request failed: {status}")));
        }
        self.handle = Some(handle);

        let runtime = unsafe { sys::bmrt_create(handle) };
        if runtime.is_null() {
            return Err(Error::BackendUnavailable(
                "bmrt_create returned null".to_string(),
            ));
        }
        self.runtime = Some(runtime);

        let loaded = unsafe { sys::bmrt_load_bmodel_data(runtime, model.as_ptr(), model.len()) };
        if !loaded {
            return Err(Error::BackendUnavailable(
                "bmrt_load_bmodel_data failed".to_string(),
            ));
        }

        self.options = sophon.clone();
        self.input_infos =
            vec![TensorInfo::new(Shape::new([1]), DataType::F32).with_layout(DataFormat::Auto)];
        self.output_infos = self.input_infos.clone();
        Ok(())
    }

    fn reshape(&mut self, input_shapes: &[Shape]) -> Result<()> {
        if input_shapes.is_empty() {
            return Err(Error::InvalidOption(
                "Sophon reshape requires at least one input shape".to_string(),
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
                "Sophon backend requires at least one input tensor".to_string(),
            ));
        }
        Err(Error::BackendUnavailable(
            "Sophon FFI execution path requires SDK validation".to_string(),
        ))
    }
}

impl Drop for SophonBackend {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            unsafe {
                sys::bmrt_destroy(runtime);
            }
        }
        if let Some(handle) = self.handle.take() {
            unsafe {
                sys::bm_dev_free(handle);
            }
        }
    }
}

fn create_sophon_backend() -> Box<dyn InferBackend> {
    Box::new(SophonBackend::new())
}

inventory::submit! {
    BackendDescriptor {
        kind: BackendKind::Sophon,
        name: "sophon",
        create: create_sophon_backend,
    }
}
