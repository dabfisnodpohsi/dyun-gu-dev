use crate::{Error, Result, RuntimeOption, TensorInfo};

/// Backend families available to the runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Mock,
    OpenVINO,
}

/// A backend implementation.
pub trait InferBackend: Send {
    fn kind(&self) -> BackendKind;
    fn init(&mut self, option: &RuntimeOption) -> Result<()>;
    fn reshape(&mut self, input_shapes: &[dg_core::Shape]) -> Result<()>;
    fn input_count(&self) -> usize;
    fn output_count(&self) -> usize;
    fn input_info(&self, index: usize) -> Result<&TensorInfo>;
    fn output_info(&self, index: usize) -> Result<&TensorInfo>;
    fn input_infos(&self) -> &[TensorInfo];
    fn output_infos(&self) -> &[TensorInfo];
    fn run(&mut self, inputs: &[dg_core::Tensor]) -> Result<Vec<dg_core::Tensor>>;
}

/// Static backend descriptor used by the registry.
pub struct BackendDescriptor {
    pub kind: BackendKind,
    pub name: &'static str,
    pub create: fn() -> Box<dyn InferBackend>,
}

/// Discover registered backends.
pub fn registered_backends() -> Vec<&'static BackendDescriptor> {
    inventory::iter::<BackendDescriptor>.into_iter().collect()
}

/// Construct a backend by kind.
pub fn create_backend(kind: BackendKind) -> Result<Box<dyn InferBackend>> {
    registered_backends()
        .into_iter()
        .find(|descriptor| descriptor.kind == kind)
        .map(|descriptor| (descriptor.create)())
        .ok_or(Error::UnsupportedBackend(kind))
}
