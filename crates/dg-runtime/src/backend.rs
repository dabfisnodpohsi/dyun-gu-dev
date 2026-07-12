use crate::{BackendConfig, Error, Result, RuntimeOption, TensorInfo};

/// Backend families available to the runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Mock,
    OpenVINO,
    Rknn,
    TensorRt,
    Sophon,
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

    fn run_with_stream(
        &mut self,
        inputs: &[dg_core::Tensor],
        _stream: Option<&dyn dg_core::Stream>,
    ) -> Result<Vec<dg_core::Tensor>> {
        self.run(inputs)
    }
}

/// Static backend descriptor used by the registry.
pub struct BackendDescriptor {
    pub kind: BackendKind,
    pub name: &'static str,
    pub create: fn() -> Box<dyn InferBackend>,
    pub configure: fn(BackendConfig) -> Result<RuntimeOption>,
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

/// Build runtime options through the backend registered under `name`.
pub fn configure_backend(name: &str, config: BackendConfig) -> Result<RuntimeOption> {
    registered_backends()
        .into_iter()
        .find(|descriptor| descriptor.name == name)
        .ok_or_else(|| Error::UnsupportedBackendName(name.to_string()))
        .and_then(|descriptor| (descriptor.configure)(config))
}
