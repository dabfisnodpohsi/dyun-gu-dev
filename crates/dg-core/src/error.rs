use thiserror::Error;

use crate::DeviceKind;

/// Result type used throughout `dg-core`.
pub type Result<T> = core::result::Result<T, Error>;

/// Error categories shared by the core abstractions.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("device error: {0}")]
    Device(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("media error: {0}")]
    Media(String),
    #[error("I/O error: {0}")]
    Io(String),
    #[error("shape error: {0}")]
    Shape(String),
    #[error("quantization error: {0}")]
    Quantization(String),
    #[error("tensor error: {0}")]
    Tensor(String),
    #[error("buffer error: {0}")]
    Buffer(String),
    #[error("stream error: {0}")]
    Stream(String),
    #[error("event error: {0}")]
    Event(String),
    #[error("out of memory")]
    OutOfMemory,
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("unsupported device: {0:?}")]
    UnsupportedDevice(DeviceKind),
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}
