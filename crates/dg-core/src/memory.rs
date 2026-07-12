use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{Buffer, BufferDesc, DeviceKind, Error, Result};

type ReleaseFn = Box<dyn FnMut() + Send + 'static>;
type ReleaseCell = Arc<Mutex<Option<ReleaseFn>>>;

/// Framework-level memory domains used for zero-copy planning and external buffer import.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryDomain {
    Host,
    DmaBuf,
    DrmPrime,
    VaapiSurface,
    CudaDevice,
    MppBuffer,
    SophonDevice,
    Opaque,
}

/// External ownership metadata carried alongside imported buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExternalHandle {
    pub fd: Option<i32>,
    pub raw: u64,
}

impl ExternalHandle {
    pub const fn none() -> Self {
        Self { fd: None, raw: 0 }
    }

    pub const fn from_fd(fd: i32) -> Self {
        Self {
            fd: Some(fd),
            raw: 0,
        }
    }

    pub const fn from_raw(raw: u64) -> Self {
        Self { fd: None, raw }
    }
}

/// RAII drop guard used to release imported external ownership exactly once.
///
/// The guard stores a boxed callback and calls it when the final guard reference is dropped.
/// This keeps the core crate free of `unsafe` while still allowing imported resources to own
/// their lifetime separately from the logical buffer handle.
pub struct ExternalDropGuard {
    callback: ReleaseCell,
}

impl ExternalDropGuard {
    pub fn new(release: impl FnOnce() + Send + 'static) -> Self {
        let mut release = Some(release);
        let callback: ReleaseFn = Box::new(move || {
            if let Some(release) = release.take() {
                release();
            }
        });
        Self {
            callback: Arc::new(Mutex::new(Some(callback))),
        }
    }
}

impl Drop for ExternalDropGuard {
    fn drop(&mut self) {
        if let Ok(mut callback) = self.callback.lock() {
            if let Some(mut release) = callback.take() {
                release();
            }
        }
    }
}

impl core::fmt::Debug for ExternalDropGuard {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExternalDropGuard").finish_non_exhaustive()
    }
}

/// Memory allocation interface used by devices and reusable pools.
pub trait Allocator: Send + Sync {
    fn allocate(&self, desc: BufferDesc) -> Result<Buffer>;
    fn deallocate(&self, buffer: Buffer) -> Result<()>;
}

/// CPU allocator backed by ordinary host memory.
#[derive(Debug, Default)]
pub struct CpuAllocator;

impl Allocator for CpuAllocator {
    fn allocate(&self, desc: BufferDesc) -> Result<Buffer> {
        if desc.align == 0 {
            return Err(Error::InvalidArgument(
                "buffer alignment must be non-zero".to_string(),
            ));
        }
        Ok(Buffer::new_host(DeviceKind::Cpu, desc))
    }

    fn deallocate(&self, _buffer: Buffer) -> Result<()> {
        Ok(())
    }
}

/// Reusable allocation pool keyed by buffer size and alignment.
pub struct MemoryPool {
    allocator: Arc<dyn Allocator>,
    buffers: Mutex<HashMap<(usize, usize), Vec<Buffer>>>,
}

impl core::fmt::Debug for MemoryPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MemoryPool")
            .field("cached_buffer_count", &self.cached_buffer_count())
            .finish_non_exhaustive()
    }
}

impl MemoryPool {
    pub fn new(allocator: Arc<dyn Allocator>) -> Self {
        Self {
            allocator,
            buffers: Mutex::new(HashMap::new()),
        }
    }

    pub fn cached_buffer_count(&self) -> usize {
        match self.buffers.lock() {
            Ok(buffers) => buffers.values().map(Vec::len).sum(),
            Err(poisoned) => poisoned.into_inner().values().map(Vec::len).sum(),
        }
    }

    fn take_cached(&self, desc: BufferDesc) -> Option<Buffer> {
        let mut buffers = match self.buffers.lock() {
            Ok(buffers) => buffers,
            Err(poisoned) => poisoned.into_inner(),
        };
        let cached = buffers.get_mut(&(desc.size_bytes, desc.align))?;
        let buffer = cached.pop();
        if cached.is_empty() {
            buffers.remove(&(desc.size_bytes, desc.align));
        }
        buffer
    }
}

impl Allocator for MemoryPool {
    fn allocate(&self, desc: BufferDesc) -> Result<Buffer> {
        if desc.align == 0 {
            return Err(Error::InvalidArgument(
                "buffer alignment must be non-zero".to_string(),
            ));
        }
        self.take_cached(desc)
            .map_or_else(|| self.allocator.allocate(desc), Ok)
    }

    fn deallocate(&self, buffer: Buffer) -> Result<()> {
        if buffer.domain() != MemoryDomain::Host || buffer.ref_count() != 1 {
            return self.allocator.deallocate(buffer);
        }
        let desc = buffer.desc();
        let mut buffers = match self.buffers.lock() {
            Ok(buffers) => buffers,
            Err(poisoned) => poisoned.into_inner(),
        };
        buffers
            .entry((desc.size_bytes, desc.align))
            .or_default()
            .push(buffer);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Allocator, CpuAllocator, MemoryPool};
    use crate::BufferDesc;
    use std::sync::Arc;

    #[test]
    fn memory_pool_reuses_matching_buffers() {
        let pool = MemoryPool::new(Arc::new(CpuAllocator));
        let first = pool
            .allocate(BufferDesc::new(64, 16))
            .expect("allocate first buffer");
        pool.deallocate(first).expect("deallocate first buffer");
        assert_eq!(pool.cached_buffer_count(), 1);

        let second = pool
            .allocate(BufferDesc::new(64, 16))
            .expect("allocate second buffer");
        assert_eq!(pool.cached_buffer_count(), 0);
        assert_eq!(second.desc(), BufferDesc::new(64, 16));
    }

    #[test]
    fn memory_pool_does_not_reuse_different_descriptors() {
        let pool = MemoryPool::new(Arc::new(CpuAllocator));
        let first = pool
            .allocate(BufferDesc::new(64, 16))
            .expect("allocate first buffer");
        pool.deallocate(first).expect("deallocate first buffer");

        let second = pool
            .allocate(BufferDesc::new(32, 16))
            .expect("allocate second buffer");
        assert_eq!(second.desc(), BufferDesc::new(32, 16));
        assert_eq!(pool.cached_buffer_count(), 1);
    }
}
