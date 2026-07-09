use std::path::PathBuf;

use dg_core::{DataType, DeviceKind};

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
}

impl BackendOptions {
    pub fn as_mock(&self) -> Option<&MockOptions> {
        match self {
            Self::Mock(options) => Some(options),
            Self::OpenVINO(_) => None,
        }
    }

    pub fn as_openvino(&self) -> Option<&OpenVINOOptions> {
        match self {
            Self::Mock(_) => None,
            Self::OpenVINO(options) => Some(options),
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

/// Unified runtime configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeOption {
    pub backend: crate::backend::BackendKind,
    pub model_source: ModelSource,
    pub precision: Option<DataType>,
    pub device: Option<DeviceKind>,
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

    pub fn with_core_mask(mut self, core_mask: u32) -> Self {
        self.core_mask = Some(core_mask);
        self
    }
}
