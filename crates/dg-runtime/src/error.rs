use dg_core::{DeployMode, DeviceKind};
use thiserror::Error;

/// Result type used by runtime abstractions.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors surfaced by runtime backends and backend selection.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    #[error("{0}")]
    Core(#[from] dg_core::Error),
    #[error("invalid option: {0}")]
    InvalidOption(String),
    #[error("unsupported backend: {0:?}")]
    UnsupportedBackend(crate::backend::BackendKind),
    #[error("unsupported precision: {0:?}")]
    UnsupportedPrecision(dg_core::DataType),
    #[error("unsupported device: {0:?}")]
    UnsupportedDevice(DeviceKind),
    #[error("unsupported deployment mode: {0:?}")]
    UnsupportedDeployment(DeployMode),
    #[error("unsupported model source: {0}")]
    UnsupportedModelSource(String),
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("IO error: {0}")]
    Io(String),
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}
