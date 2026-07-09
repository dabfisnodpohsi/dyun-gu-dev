use thiserror::Error;

/// Result type used by stream abstractions.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors surfaced by stream adapters and in-memory tests.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("stream closed")]
    Closed,
    #[error("end of stream")]
    EndOfStream,
    #[error("buffer error: {0}")]
    Buffer(String),
    #[error("media error: {0}")]
    Media(String),
    #[error("sdk error: {0}")]
    Sdk(String),
    #[error("runtime error: {0}")]
    Runtime(String),
}

impl From<dg_core::Error> for Error {
    fn from(value: dg_core::Error) -> Self {
        Self::Buffer(value.to_string())
    }
}
