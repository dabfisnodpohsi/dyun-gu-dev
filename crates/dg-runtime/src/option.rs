use std::path::PathBuf;

use dg_core::{DataType, DeployMode, DeviceKind, StreamKind};
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::{mock::MockOptions, Error, Result};

/// Model payload for backends that ingest files or bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelSource {
    File(PathBuf),
    Bytes(Vec<u8>),
}

/// Backend-agnostic model representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelFormat {
    Auto,
    Onnx,
    Engine,
    Rknn,
    Bmodel,
    OpenvinoIr,
}

impl ModelFormat {
    /// Infers a model format from a file source, or returns `Auto` otherwise.
    pub fn from_source(source: &ModelSource) -> Self {
        let ModelSource::File(path) = source else {
            return Self::Auto;
        };
        let Some(extension) = path.extension() else {
            return Self::Auto;
        };
        match extension.to_string_lossy().to_ascii_lowercase().as_str() {
            "onnx" => Self::Onnx,
            "engine" | "plan" => Self::Engine,
            "rknn" => Self::Rknn,
            "bmodel" => Self::Bmodel,
            "xml" => Self::OpenvinoIr,
            _ => Self::Auto,
        }
    }
}

/// Backend-agnostic device core selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoreSelection {
    #[default]
    Auto,
    Single(u8),
    Mask(u32),
    All,
}

/// Opaque stream handle for backend-specific interpretation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExternalStreamHandle {
    pub kind: StreamKind,
    pub raw: u64,
}

/// Backend-specific options grouped by backend family.
#[derive(Clone, Debug, PartialEq)]
pub enum BackendOptions {
    Mock(MockOptions),
    OpenVINO(OpenVINOOptions),
    Rknn(RknnOptions),
    TensorRt(TensorRtOptions),
    Sophon(SophonOptions),
}

/// Common GraphSpec configuration delegated to a registered backend.
#[derive(Clone, Debug)]
pub struct BackendConfig {
    model: Option<PathBuf>,
    precision: Option<DataType>,
    device: Option<DeviceKind>,
    deploy_mode: Option<DeployMode>,
    core_mask: Option<u32>,
    options: Value,
}

impl BackendConfig {
    pub fn new(model: Option<PathBuf>, options: Value) -> Self {
        Self {
            model,
            precision: None,
            device: None,
            deploy_mode: None,
            core_mask: None,
            options,
        }
    }

    pub fn with_precision(mut self, precision: DataType) -> Self {
        self.precision = Some(precision);
        self
    }

    pub fn with_device(mut self, device: DeviceKind) -> Self {
        self.device = Some(device);
        self
    }

    pub fn with_deploy_mode(mut self, deploy_mode: DeployMode) -> Self {
        self.deploy_mode = Some(deploy_mode);
        self
    }

    pub fn with_core_mask(mut self, core_mask: u32) -> Self {
        self.core_mask = Some(core_mask);
        self
    }

    pub fn deploy_mode(&self) -> Option<DeployMode> {
        self.deploy_mode
    }

    pub fn core_mask(&self) -> Option<u32> {
        self.core_mask
    }

    pub fn parse_options<T: DeserializeOwned>(&self, backend: &str) -> Result<T> {
        let value = if self.options.is_null() {
            Value::Object(serde_json::Map::new())
        } else {
            self.options.clone()
        };
        serde_json::from_value(value)
            .map_err(|err| Error::InvalidOption(format!("{backend} options: {err}")))
    }

    pub fn require_model_file(&self, backend: &str) -> Result<ModelSource> {
        self.model
            .clone()
            .map(ModelSource::File)
            .ok_or_else(|| Error::InvalidOption(format!("{backend} requires a model file path")))
    }

    pub fn into_runtime_option(
        self,
        backend: crate::backend::BackendKind,
        model_source: ModelSource,
        backend_options: BackendOptions,
    ) -> RuntimeOption {
        RuntimeOption {
            backend,
            model_source,
            precision: self.precision,
            device: self.device,
            deploy_mode: self.deploy_mode,
            core_mask: self.core_mask,
            device_id: None,
            core: CoreSelection::Auto,
            cpu_threads: None,
            model_format: ModelFormat::Auto,
            zero_copy: false,
            dynamic_shape: false,
            external_stream: None,
            backend_options,
        }
    }
}

impl BackendOptions {
    pub fn as_mock(&self) -> Option<&MockOptions> {
        match self {
            Self::Mock(options) => Some(options),
            Self::OpenVINO(_) | Self::Rknn(_) | Self::TensorRt(_) | Self::Sophon(_) => None,
        }
    }

    pub fn as_openvino(&self) -> Option<&OpenVINOOptions> {
        match self {
            Self::Mock(_) | Self::Rknn(_) | Self::TensorRt(_) | Self::Sophon(_) => None,
            Self::OpenVINO(options) => Some(options),
        }
    }

    pub fn as_rknn(&self) -> Option<&RknnOptions> {
        match self {
            Self::Mock(_) | Self::OpenVINO(_) | Self::TensorRt(_) | Self::Sophon(_) => None,
            Self::Rknn(options) => Some(options),
        }
    }

    pub fn as_tensorrt(&self) -> Option<&TensorRtOptions> {
        match self {
            Self::TensorRt(options) => Some(options),
            Self::Mock(_) | Self::OpenVINO(_) | Self::Rknn(_) | Self::Sophon(_) => None,
        }
    }

    pub fn as_sophon(&self) -> Option<&SophonOptions> {
        match self {
            Self::Sophon(options) => Some(options),
            Self::Mock(_) | Self::OpenVINO(_) | Self::Rknn(_) | Self::TensorRt(_) => None,
        }
    }
}

/// Options passed to the OpenVINO backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenVINOOptions {
    pub device: String,
}

impl Default for OpenVINOOptions {
    fn default() -> Self {
        Self {
            device: "CPU".to_string(),
        }
    }
}

/// Options passed to the RKNN backend.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RknnOptions {
    pub core_mask: Option<u32>,
    pub enable_zero_copy: bool,
    pub dynamic_shape: bool,
}

/// Options passed to the TensorRT backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRtOptions {
    pub device_id: Option<u32>,
    pub workspace_size_mb: usize,
    pub enable_fp16: bool,
    pub enable_int8: bool,
}

impl Default for TensorRtOptions {
    fn default() -> Self {
        Self {
            device_id: None,
            workspace_size_mb: 1024,
            enable_fp16: false,
            enable_int8: false,
        }
    }
}

/// Options passed to the Sophon backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SophonOptions {
    pub deploy_mode: DeployMode,
    pub device_id: Option<u32>,
    pub core_mask: Option<u32>,
}

impl Default for SophonOptions {
    fn default() -> Self {
        Self {
            deploy_mode: DeployMode::Host,
            device_id: None,
            core_mask: None,
        }
    }
}

/// Unified runtime configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeOption {
    pub backend: crate::backend::BackendKind,
    pub model_source: ModelSource,
    pub precision: Option<DataType>,
    pub device: Option<DeviceKind>,
    pub deploy_mode: Option<DeployMode>,
    pub core_mask: Option<u32>,
    pub device_id: Option<u16>,
    pub core: CoreSelection,
    pub cpu_threads: Option<usize>,
    pub model_format: ModelFormat,
    pub zero_copy: bool,
    pub dynamic_shape: bool,
    pub external_stream: Option<ExternalStreamHandle>,
    pub backend_options: BackendOptions,
}

impl RuntimeOption {
    pub fn new(
        backend: crate::backend::BackendKind,
        model_source: ModelSource,
        backend_options: BackendOptions,
    ) -> Self {
        Self {
            backend,
            model_source,
            precision: None,
            device: None,
            deploy_mode: None,
            core_mask: None,
            device_id: None,
            core: CoreSelection::Auto,
            cpu_threads: None,
            model_format: ModelFormat::Auto,
            zero_copy: false,
            dynamic_shape: false,
            external_stream: None,
            backend_options,
        }
    }

    pub fn with_precision(mut self, precision: DataType) -> Self {
        self.precision = Some(precision);
        self
    }

    pub fn with_device(mut self, device: DeviceKind) -> Self {
        self.device = Some(device);
        self
    }

    pub fn with_deploy_mode(mut self, deploy_mode: DeployMode) -> Self {
        self.deploy_mode = Some(deploy_mode);
        self
    }

    pub fn with_core_mask(mut self, core_mask: u32) -> Self {
        self.core_mask = Some(core_mask);
        self
    }

    pub fn with_device_id(mut self, device_id: u16) -> Self {
        self.device_id = Some(device_id);
        self
    }

    pub fn with_core(mut self, core: CoreSelection) -> Self {
        self.core = core;
        self
    }

    pub fn with_cpu_threads(mut self, cpu_threads: usize) -> Self {
        self.cpu_threads = Some(cpu_threads);
        self
    }

    pub fn with_model_format(mut self, model_format: ModelFormat) -> Self {
        self.model_format = model_format;
        self
    }

    pub fn with_zero_copy(mut self, zero_copy: bool) -> Self {
        self.zero_copy = zero_copy;
        self
    }

    pub fn with_dynamic_shape(mut self, dynamic_shape: bool) -> Self {
        self.dynamic_shape = dynamic_shape;
        self
    }

    pub fn with_external_stream(mut self, external_stream: ExternalStreamHandle) -> Self {
        self.external_stream = Some(external_stream);
        self
    }
}
