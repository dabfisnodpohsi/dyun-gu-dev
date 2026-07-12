#![forbid(unsafe_code)]

//! OpenVINO backend integration.
//!
//! This crate intentionally ships as a feature-gated adapter for M1:
//! - the default build is a no-op placeholder that does not require OpenVINO;
//! - the `backend` feature enables the real implementation through the
//!   `dg-openvino-sys` FFI/link boundary around the community `openvino` crate.

#[cfg(feature = "backend")]
mod backend;
mod binding;

pub use binding::{select_external_binding_path, ExternalBindingPath};

#[cfg(feature = "backend")]
pub use backend::{backend_enabled, OpenVINOBackend, OpenVINOOptions};

#[cfg(not(feature = "backend"))]
/// Returns `false` when the real OpenVINO integration is not compiled in.
pub const fn backend_enabled() -> bool {
    false
}
