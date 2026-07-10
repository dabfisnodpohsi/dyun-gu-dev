//! TensorRT backend integration.
//!
//! The default build is a no-op placeholder so the workspace stays green on
//! machines without vendor SDKs. Enabling the `backend` feature switches the
//! crate to the real TensorRT runtime adapter and requires `TENSORRT_ROOT` to
//! point at a local SDK installation.

#[cfg(any(feature = "backend", test))]
mod backend;
#[cfg(any(feature = "backend", test))]
mod convert;
#[cfg(all(test, not(feature = "backend")))]
mod mock_sys;

pub use dg_runtime::TensorRtOptions;

#[cfg(feature = "backend")]
pub use backend::{backend_enabled, TensorRtBackend};

#[cfg(not(feature = "backend"))]
/// Returns `false` when the real TensorRT runtime is not compiled in.
pub const fn backend_enabled() -> bool {
    false
}
