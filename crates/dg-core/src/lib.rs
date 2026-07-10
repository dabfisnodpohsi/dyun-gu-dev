#![forbid(unsafe_code)]

//! Core inference abstractions for the M0 workspace.
//!
//! This crate provides the device, buffer, tensor, datatype, quantization,
//! stream, and event building blocks that the later runtime and graph layers
//! will compose.

mod buffer;
mod datatype;
mod deployment;
mod detection;
mod device;
mod error;
mod format;
mod memory;
mod quantization;
mod shape;
mod stream;
mod tensor;

pub use buffer::{Buffer, BufferDesc};
pub use datatype::{
    pack_float4, pack_int4, unpack_float4, unpack_int4, DataType, NativeDataType, TypeCode,
};
pub use deployment::DeployMode;
pub use detection::{BBox, Detection};
pub use device::{CpuDevice, Device, DeviceKind, MemoryType};
pub use error::{Error, Result};
pub use format::DataFormat;
pub use memory::{ExternalDropGuard, ExternalHandle, MemoryDomain};
pub use quantization::{Quantization, QuantizationScheme};
pub use shape::{Shape, Strides};
pub use stream::{CpuEvent, CpuStream, Event, EventKind, Stream, StreamKind};
pub use tensor::{Tensor, TensorDesc};
