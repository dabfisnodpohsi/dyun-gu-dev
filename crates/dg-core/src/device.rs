use crate::{Buffer, BufferDesc, CpuStream, Error, Result, Stream};

/// Device family identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    IntelGpu,
    IntelNpu,
    CudaGpu,
    RknnNpu,
    SophonTpu,
}

/// Memory ownership model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryType {
    Host,
    Device,
    Unified,
}

/// Device abstraction used by runtime and graph layers.
pub trait Device: Send + Sync {
    fn kind(&self) -> DeviceKind;
    fn name(&self) -> &str;
    fn alloc(&self, desc: BufferDesc) -> Result<Buffer>;
    fn free(&self, buffer: Buffer) -> Result<()>;
    fn memcpy_h2d(&self, dst: &Buffer, src: &[u8]) -> Result<()>;
    fn memcpy_d2h(&self, src: &Buffer, dst: &mut [u8]) -> Result<()>;
    fn memcpy_d2d(&self, dst: &Buffer, src: &Buffer) -> Result<()>;
    fn create_stream(&self) -> Result<Box<dyn Stream>>;
    fn synchronize(&self) -> Result<()>;
}

/// CPU device backed by ordinary heap memory.
#[derive(Clone, Debug, Default)]
pub struct CpuDevice;

impl CpuDevice {
    pub fn new() -> Self {
        Self
    }
}

impl Device for CpuDevice {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn name(&self) -> &str {
        "cpu"
    }

    fn alloc(&self, desc: BufferDesc) -> Result<Buffer> {
        if desc.align == 0 {
            return Err(Error::InvalidArgument(
                "buffer alignment must be non-zero".to_string(),
            ));
        }
        Ok(Buffer::new_host(self.kind(), desc))
    }

    fn free(&self, _buffer: Buffer) -> Result<()> {
        Ok(())
    }

    fn memcpy_h2d(&self, dst: &Buffer, src: &[u8]) -> Result<()> {
        dst.write_from_slice(src)
    }

    fn memcpy_d2h(&self, src: &Buffer, dst: &mut [u8]) -> Result<()> {
        src.copy_into(dst)
    }

    fn memcpy_d2d(&self, dst: &Buffer, src: &Buffer) -> Result<()> {
        src.copy_to(dst)
    }

    fn create_stream(&self) -> Result<Box<dyn Stream>> {
        Ok(Box::new(CpuStream))
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
