#![forbid(unsafe_code)]

#[cfg(feature = "avcodec")]
pub use avcodec::core::*;

#[cfg(feature = "avcodec")]
pub use avcodec::native_free_software_registry_builder;
