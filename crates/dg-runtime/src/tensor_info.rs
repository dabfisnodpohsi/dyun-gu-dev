use dg_core::{
    DataFormat, DataType, Device, DeviceKind, Quantization, QuantizationScheme,
    Result as CoreResult, Shape, Strides, Tensor, TensorDesc,
};

/// Static tensor metadata queried from a backend.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub name: Option<String>,
    pub shape: Shape,
    pub dtype: DataType,
    pub layout: Option<DataFormat>,
    pub quant: Quantization,
    /// Element strides for padded layouts (e.g. RKNN `w_stride`).
    pub strides: Option<Strides>,
    /// Physical byte size including stride padding (RKNN `size_with_stride`).
    pub size_with_stride: Option<usize>,
}

impl TensorInfo {
    pub fn new(shape: Shape, dtype: DataType) -> Self {
        Self {
            name: None,
            shape,
            dtype,
            layout: None,
            quant: Quantization::none(),
            strides: None,
            size_with_stride: None,
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

    pub fn with_quantization(mut self, quant: Quantization) -> Self {
        self.quant = quant;
        self
    }

    pub fn with_strides(mut self, strides: Strides) -> Self {
        self.strides = Some(strides);
        self
    }

    pub fn with_size_with_stride(mut self, size_with_stride: usize) -> Self {
        self.size_with_stride = Some(size_with_stride);
        self
    }

    pub fn is_quantized(&self) -> bool {
        self.quant.scheme != QuantizationScheme::None
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
        if let Some(strides) = &self.strides {
            desc = desc.with_strides(strides.clone());
        }
        desc = desc.with_quantization(self.quant.clone());
        desc
    }

    pub fn allocate(&self, device: &dyn Device) -> CoreResult<Tensor> {
        Tensor::allocate(device, self.tensor_desc(device.kind()))
    }
}
