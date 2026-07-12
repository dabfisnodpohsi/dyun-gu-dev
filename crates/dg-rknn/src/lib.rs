//! RKNN backend integration.
//!
//! The default build is a no-op placeholder so the workspace stays green on
//! machines without vendor SDKs. Enabling the `backend` feature switches the
//! crate to the real RKNN runtime adapter through `dg-rknn-sys`, which requires
//! `RKNN_SDK_ROOT` to point at a local SDK installation.

#[cfg(any(feature = "backend", test))]
mod backend;
mod io;
#[cfg(all(test, not(feature = "backend")))]
mod mock_sys;

pub use dg_runtime::RknnOptions;
pub use io::{
    depad_bytes, pad_bytes, padded_byte_len, quantization_from_rknn, select_io_path,
    strides_from_w_stride, IoPath,
};

#[cfg(feature = "backend")]
pub use backend::{backend_enabled, RknnBackend};

#[cfg(not(feature = "backend"))]
/// Returns `false` when the real RKNN runtime is not compiled in.
pub const fn backend_enabled() -> bool {
    false
}
