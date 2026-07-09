use std::path::PathBuf;

use dg_core::{DataType, DeployMode, DeviceKind};

use crate::mock::MockOptions;

/// Model payload for backends that ingest files or bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelSource {
    File(PathBuf),
    Bytes(Vec<u8>),
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
}
