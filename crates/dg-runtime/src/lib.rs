#![forbid(unsafe_code)]

//! Runtime abstractions for inference backends.
//!
//! `dg-runtime` owns the backend trait, registration registry, runtime options,
//! and the built-in mock backend used by CI and end-to-end integration tests.

mod backend;
mod capabilities;
mod error;
mod mock;
mod option;
mod runtime;
mod tensor_info;

pub use backend::{
    create_backend, registered_backends, BackendDescriptor, BackendKind, InferBackend,
};
pub use capabilities::{
    backend_capabilities, supports_deployment, supports_device, supports_precision,
    BackendCapabilities,
};
pub use error::{Error, Result};
pub use mock::MockOptions;
pub use option::{
    BackendOptions, ModelSource, OpenVINOOptions, RknnOptions, RuntimeOption, SophonOptions,
    TensorRtOptions,
};
pub use runtime::Runtime;
pub use tensor_info::TensorInfo;

inventory::collect!(BackendDescriptor);
