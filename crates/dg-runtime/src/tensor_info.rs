use dg_core::{
    DataFormat, DataType, Device, DeviceKind, Result as CoreResult, Shape, Tensor, TensorDesc,
};

/// Static tensor metadata queried from a backend.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub name: Option<String>,
    pub shape: Shape,
    pub dtype: DataType,
    pub layout: Option<DataFormat>,
}

impl TensorInfo {
    pub fn new(shape: Shape, dtype: DataType) -> Self {
        Self {
            name: None,
            shape,
            dtype,
            layout: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_layout(mut self, layout: DataFormat) -> Self {
        self.layout = Some(layout);
        self
    }

    pub fn with_shape(mut self, shape: Shape) -> Self {
        self.shape = shape;
        self
    }

    pub fn tensor_desc(&self, device: DeviceKind) -> TensorDesc {
        let mut desc = TensorDesc::new(
            self.shape.clone(),
            self.dtype,
            self.layout.unwrap_or(DataFormat::Auto),
            device,
        );
        if let Some(name) = &self.name {
            desc = desc.with_name(name.clone());
        }
        desc
    }

    pub fn allocate(&self, device: &dyn Device) -> CoreResult<Tensor> {
        Tensor::allocate(device, self.tensor_desc(device.kind()))
    }
}
