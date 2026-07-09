use tracing::trace;

use crate::{
    Buffer, BufferDesc, DataFormat, DataType, Device, DeviceKind, Error, Quantization, Result,
    Shape, Strides,
};

/// Tensor descriptor with layout metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorDesc {
    name: Option<String>,
    shape: Shape,
    strides: Option<Strides>,
    dtype: DataType,
    format: DataFormat,
    quant: Quantization,
    device: DeviceKind,
}

impl TensorDesc {
    pub fn new(shape: Shape, dtype: DataType, format: DataFormat, device: DeviceKind) -> Self {
        Self {
            name: None,
            shape,
            strides: None,
            dtype,
            format,
            quant: Quantization::default(),
            device,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_strides(mut self, strides: Strides) -> Self {
        self.strides = Some(strides);
        self
    }

    pub fn with_quantization(mut self, quant: Quantization) -> Self {
        self.quant = quant;
        self
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn strides(&self) -> Option<&Strides> {
        self.strides.as_ref()
    }

    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    pub fn format(&self) -> DataFormat {
        self.format
    }

    pub fn quantization(&self) -> &Quantization {
        &self.quant
    }

    pub fn device(&self) -> DeviceKind {
        self.device
    }

    pub fn element_count(&self) -> Result<usize> {
        self.shape.element_count()
    }

    pub fn storage_bytes(&self) -> Result<usize> {
        self.dtype.storage_bytes_for_shape(&self.shape)
    }
}

/// Tensor value backed by a shared buffer.
#[derive(Clone, Debug)]
pub struct Tensor {
    desc: TensorDesc,
    buffer: Buffer,
}

impl Tensor {
    pub fn allocate(device: &dyn Device, desc: TensorDesc) -> Result<Self> {
        trace!(device = device.name(), "allocating tensor");
        let buffer = device.alloc(BufferDesc::new(desc.storage_bytes()?, 1))?;
        if buffer.device() != desc.device() {
            return Err(Error::Tensor(
                "allocated buffer device does not match tensor device".to_string(),
            ));
        }
        Ok(Self { desc, buffer })
    }

    pub fn from_buffer(desc: TensorDesc, buffer: Buffer) -> Result<Self> {
        if buffer.len() != desc.storage_bytes()? {
            return Err(Error::Tensor(
                "buffer size does not match tensor descriptor".to_string(),
            ));
        }
        if buffer.device() != desc.device() {
            return Err(Error::Tensor(
                "buffer device does not match tensor descriptor".to_string(),
            ));
        }
        Ok(Self { desc, buffer })
    }

    pub fn desc(&self) -> &TensorDesc {
        &self.desc
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn reshape(&mut self, shape: Shape) -> Result<()> {
        let old_elements = self.desc.shape.element_count()?;
        let new_elements = shape.element_count()?;
        if old_elements != new_elements {
            return Err(Error::Shape(
                "reshape must preserve logical element count".to_string(),
            ));
        }

        if self
            .desc
            .strides
            .as_ref()
            .is_some_and(|strides| !strides.is_contiguous_for(&self.desc.shape))
        {
            return Err(Error::Shape(
                "cannot reshape a tensor with non-contiguous strides".to_string(),
            ));
        }

        self.desc.shape = shape;
        self.desc.strides = Some(self.desc.shape.contiguous_strides());
        Ok(())
    }

    pub fn copy_to(&self, dst: &mut Tensor) -> Result<()> {
        trace!("copying tensor");
        if self.buffer.len() != dst.buffer.len() {
            return Err(Error::Tensor("tensor byte sizes differ".to_string()));
        }
        self.buffer.copy_to(&dst.buffer)?;
        Ok(())
    }
}
