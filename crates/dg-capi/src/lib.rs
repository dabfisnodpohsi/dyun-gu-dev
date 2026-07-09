#![forbid(unsafe_code)]

//! C API crate placeholder for M0.
//!
//! The stable opaque-pointer ABI will be implemented in a later milestone once
//! the Rust-side core abstractions settle.

/// Returns the package version for smoke testing.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
