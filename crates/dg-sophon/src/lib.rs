//! Sophon backend integration.
//!
//! The default build is a no-op placeholder so the workspace stays green on
//! machines without vendor SDKs. Enabling the `backend` feature switches the
//! crate to the real Sophon runtime adapter and requires `LIBSOPHON_ROOT` to
//! point at a local SDK installation.

#[cfg(feature = "backend")]
mod backend;

pub use dg_runtime::SophonOptions;

#[cfg(feature = "backend")]
pub use backend::{backend_enabled, SophonBackend};

#[cfg(not(feature = "backend"))]
/// Returns `false` when the real Sophon runtime is not compiled in.
pub const fn backend_enabled() -> bool {
    false
}
