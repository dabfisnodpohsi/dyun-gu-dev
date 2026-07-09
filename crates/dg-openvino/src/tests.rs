#![cfg(feature = "backend")]

use dg_core::{DataFormat, DataType, DeviceKind, Shape, Tensor, TensorDesc};
use dg_runtime::{BackendOptions, ModelSource, Runtime, RuntimeOption};
use dg_openvino::OpenVINOOptions;

#[test]
fn feature_gate_compiles_and_placeholder_is_not_used_by_default() {
    assert!(dg_openvino::backend_enabled());
}
