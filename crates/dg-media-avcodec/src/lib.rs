#![forbid(unsafe_code)]

#[cfg(feature = "avcodec")]
pub use avcodec_core::*;

#[cfg(feature = "avcodec")]
pub use avcodec_backend_jpeg::BACKEND as JPEG_BACKEND;

#[cfg(feature = "avcodec")]
pub use avcodec_backend_zune::BACKEND as ZUNE_BACKEND;
