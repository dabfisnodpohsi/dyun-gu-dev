use std::sync::Arc;

use crate::{
    Allocator, Buffer, BufferDesc, CpuAllocator, CpuEvent, CpuStream, Error, Event, MemoryPool,
    Result, Stream,
};

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
    fn create_event(&self) -> Result<Box<dyn Event>>;
    fn allocator(&self) -> Arc<dyn Allocator>;
    fn synchronize(&self) -> Result<()>;
}

/// CPU device backed by ordinary heap memory.
#[derive(Clone, Debug)]
pub struct CpuDevice {
    allocator: Arc<MemoryPool>,
}

impl CpuDevice {
    pub fn new() -> Self {
        Self {
            allocator: Arc::new(MemoryPool::new(Arc::new(CpuAllocator))),
        }
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
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
        self.allocator.allocate(desc)
    }

    fn free(&self, buffer: Buffer) -> Result<()> {
        self.allocator.deallocate(buffer)
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

    fn create_event(&self) -> Result<Box<dyn Event>> {
        Ok(Box::new(CpuEvent))
    }

    fn allocator(&self) -> Arc<dyn Allocator> {
        self.allocator.clone()
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

/// Static device descriptor used by the registry.
pub struct DeviceDescriptor {
    pub kind: DeviceKind,
    pub name: &'static str,
    pub create: fn() -> Result<Arc<dyn Device>>,
}

/// Discover registered device adapters.
pub fn registered_devices() -> Vec<&'static DeviceDescriptor> {
    inventory::iter::<DeviceDescriptor>.into_iter().collect()
}

/// List the device kinds with registered adapters.
pub fn registered_device_kinds() -> Vec<DeviceKind> {
    registered_devices()
        .into_iter()
        .map(|descriptor| descriptor.kind)
        .collect()
}

/// Resolve a device adapter by kind.
pub fn resolve_device(kind: DeviceKind) -> Result<Arc<dyn Device>> {
    let descriptor = registered_devices()
        .into_iter()
        .find(|descriptor| descriptor.kind == kind)
        .ok_or(Error::UnsupportedDevice(kind))?;
    (descriptor.create)()
        .map_err(|error| Error::Device(format!("failed to create {kind:?} device: {error}")))
}

fn create_cpu_device() -> Result<Arc<dyn Device>> {
    Ok(Arc::new(CpuDevice::new()))
}

inventory::submit! {
    DeviceDescriptor {
        kind: DeviceKind::Cpu,
        name: "cpu",
        create: create_cpu_device,
    }
}

#[cfg(test)]
mod tests {
    use super::{Device, DeviceDescriptor, DeviceKind};
    use crate::{
        Allocator, BufferDesc, Error, Event, EventKind, MemoryType, Result, Stream, StreamKind,
    };
    use std::sync::Arc;

    #[derive(Debug, Default)]
    struct MockAllocator;

    impl Allocator for MockAllocator {
        fn allocate(&self, desc: BufferDesc) -> Result<crate::Buffer> {
            if desc.align == 0 {
                return Err(Error::InvalidArgument(
                    "buffer alignment must be non-zero".to_string(),
                ));
            }
            Ok(crate::Buffer::new_host(DeviceKind::IntelGpu, desc))
        }

        fn deallocate(&self, _buffer: crate::Buffer) -> Result<()> {
            Ok(())
        }
    }

    #[derive(Debug, Default)]
    struct MockStream;

    impl Stream for MockStream {
        fn kind(&self) -> StreamKind {
            StreamKind::Cpu
        }

        fn synchronize(&self) -> Result<()> {
            Ok(())
        }
    }

    #[derive(Debug, Default)]
    struct MockEvent;

    impl Event for MockEvent {
        fn kind(&self) -> EventKind {
            EventKind::Cpu
        }

        fn record(&self, stream: &dyn Stream) -> Result<()> {
            if stream.kind() != StreamKind::Cpu {
                return Err(Error::Event("mock event requires CPU stream".to_string()));
            }
            Ok(())
        }

        fn synchronize(&self) -> Result<()> {
            Ok(())
        }
    }

    #[derive(Debug, Default)]
    struct MockDevice {
        allocator: Arc<MockAllocator>,
    }

    impl Device for MockDevice {
        fn kind(&self) -> DeviceKind {
            DeviceKind::IntelGpu
        }

        fn name(&self) -> &str {
            "mock-intel-device"
        }

        fn alloc(&self, desc: BufferDesc) -> Result<crate::Buffer> {
            self.allocator.allocate(desc)
        }

        fn free(&self, buffer: crate::Buffer) -> Result<()> {
            self.allocator.deallocate(buffer)
        }

        fn memcpy_h2d(&self, dst: &crate::Buffer, src: &[u8]) -> Result<()> {
            dst.write_from_slice(src)
        }

        fn memcpy_d2h(&self, src: &crate::Buffer, dst: &mut [u8]) -> Result<()> {
            src.copy_into(dst)
        }

        fn memcpy_d2d(&self, dst: &crate::Buffer, src: &crate::Buffer) -> Result<()> {
            src.copy_to(dst)
        }

        fn create_stream(&self) -> Result<Box<dyn Stream>> {
            Ok(Box::new(MockStream))
        }

        fn create_event(&self) -> Result<Box<dyn Event>> {
            Ok(Box::new(MockEvent))
        }

        fn allocator(&self) -> Arc<dyn Allocator> {
            self.allocator.clone()
        }

        fn synchronize(&self) -> Result<()> {
            Ok(())
        }
    }

    fn create_mock_device() -> Result<Arc<dyn Device>> {
        Ok(Arc::new(MockDevice::default()))
    }

    inventory::submit! {
        DeviceDescriptor {
            kind: DeviceKind::IntelGpu,
            name: "mock-intel-device",
            create: create_mock_device,
        }
    }

    #[test]
    fn registry_resolves_cpu_and_mock_non_cpu_device() {
        let kinds = super::registered_device_kinds();
        assert!(kinds.contains(&DeviceKind::Cpu));
        assert!(kinds.contains(&DeviceKind::IntelGpu));

        let device = super::resolve_device(DeviceKind::IntelGpu).expect("resolve mock device");
        assert_eq!(device.kind(), DeviceKind::IntelGpu);
        assert_eq!(device.name(), "mock-intel-device");
    }

    #[test]
    fn registry_device_supports_stream_event_and_allocator() {
        let device = super::resolve_device(DeviceKind::IntelGpu).expect("resolve mock device");
        let stream = device.create_stream().expect("create mock stream");
        let event = device.create_event().expect("create mock event");
        assert_eq!(stream.kind(), StreamKind::Cpu);
        assert_eq!(event.kind(), EventKind::Cpu);
        event.record(&*stream).expect("record mock event");

        let allocator = device.allocator();
        let buffer = allocator
            .allocate(BufferDesc::new(4, 1))
            .expect("allocate mock buffer");
        assert_eq!(buffer.device(), DeviceKind::IntelGpu);
        assert_eq!(buffer.memory_type(), MemoryType::Host);
        device
            .memcpy_h2d(&buffer, &[1, 2, 3, 4])
            .expect("write mock buffer");
        let mut bytes = [0; 4];
        device
            .memcpy_d2h(&buffer, &mut bytes)
            .expect("read mock buffer");
        assert_eq!(bytes, [1, 2, 3, 4]);
        allocator
            .deallocate(buffer)
            .expect("deallocate mock buffer");
    }

    #[test]
    fn registry_reports_missing_device() {
        let error = match super::resolve_device(DeviceKind::CudaGpu) {
            Ok(_) => panic!("device should be absent"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            Error::UnsupportedDevice(DeviceKind::CudaGpu)
        ));
    }
}
