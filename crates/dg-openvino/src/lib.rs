#![forbid(unsafe_code)]

//! OpenVINO backend integration.
//!
//! This crate intentionally ships as a feature-gated adapter for M1:
//! - the default build is a no-op placeholder that does not require OpenVINO;
//! - the `backend` feature enables the real implementation using the community
//!   `openvino` crate, which keeps the crate usable on Rust 1.96.1 without a
//!   bespoke `-sys` layer in this milestone.

#[cfg(feature = "backend")]
mod backend;

#[cfg(feature = "backend")]
pub use backend::{backend_enabled, OpenVINOBackend, OpenVINOOptions};

#[cfg(not(feature = "backend"))]
/// Returns `false` when the real OpenVINO integration is not compiled in.
pub const fn backend_enabled() -> bool {
    false
}
