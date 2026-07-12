use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::{
    DeviceKind, Error, ExternalDropGuard, ExternalHandle, MemoryDomain, MemoryType, Result,
};

/// Buffer descriptor used for allocations and validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferDesc {
    pub size_bytes: usize,
    pub align: usize,
}

impl BufferDesc {
    pub fn new(size_bytes: usize, align: usize) -> Self {
        Self { size_bytes, align }
    }
}

#[derive(Clone, Debug)]
enum BufferStorage {
    Host(Arc<RwLock<Vec<u8>>>),
    External {
        bytes: Option<Arc<RwLock<Vec<u8>>>>,
        _guard: Arc<ExternalDropGuard>,
    },
}

impl BufferStorage {
    fn read_bytes(&self) -> Vec<u8> {
        match self {
            Self::Host(bytes) => read_guard(bytes).clone(),
            Self::External { bytes, .. } => bytes
                .as_ref()
                .map_or_else(Vec::new, |bytes| read_guard(bytes).clone()),
        }
    }

    fn write_from_slice(&self, src: &[u8]) -> Result<()> {
        match self {
            Self::Host(bytes) => {
                let mut guard = write_guard(bytes);
                if guard.len() != src.len() {
                    return Err(Error::Buffer(
                        "source and destination size differ".to_string(),
                    ));
                }
                guard.copy_from_slice(src);
                Ok(())
            }
            Self::External {
                bytes: Some(bytes), ..
            } => {
                let mut guard = write_guard(bytes);
                if guard.len() != src.len() {
                    return Err(Error::Buffer(
                        "source and destination size differ".to_string(),
                    ));
                }
                guard.copy_from_slice(src);
                Ok(())
            }
            Self::External { bytes: None, .. } => Err(Error::Buffer(
                "external buffer is not host-mapped; call map or stage explicitly".to_string(),
            )),
        }
    }
}

fn read_guard(lock: &RwLock<Vec<u8>>) -> RwLockReadGuard<'_, Vec<u8>> {
    match lock.read() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn write_guard(lock: &RwLock<Vec<u8>>) -> RwLockWriteGuard<'_, Vec<u8>> {
    match lock.write() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Shared byte storage with RAII semantics.
#[derive(Clone, Debug)]
pub struct Buffer {
    device: DeviceKind,
    domain: MemoryDomain,
    desc: BufferDesc,
    external: ExternalHandle,
    storage: Arc<BufferStorage>,
}

impl Buffer {
    pub(crate) fn new_host(device: DeviceKind, desc: BufferDesc) -> Self {
        Self {
            device,
            domain: MemoryDomain::Host,
            desc,
            external: ExternalHandle::none(),
            storage: Arc::new(BufferStorage::Host(Arc::new(RwLock::new(vec![
                0;
                desc.size_bytes
            ])))),
        }
    }

    pub fn allocate_host(device: DeviceKind, size_bytes: usize) -> Self {
        Self::new_host(device, BufferDesc::new(size_bytes, 1))
    }

    pub fn from_host_bytes(device: DeviceKind, desc: BufferDesc, bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() != desc.size_bytes {
            return Err(Error::Buffer(
                "host bytes do not match descriptor size".to_string(),
            ));
        }
        Ok(Self {
            device,
            domain: MemoryDomain::Host,
            desc,
            external: ExternalHandle::none(),
            storage: Arc::new(BufferStorage::Host(Arc::new(RwLock::new(bytes)))),
        })
    }

    pub fn from_external_with_host_bytes(
        device: DeviceKind,
        domain: MemoryDomain,
        desc: BufferDesc,
        external: ExternalHandle,
        bytes: Vec<u8>,
        guard: ExternalDropGuard,
    ) -> Result<Self> {
        if bytes.len() != desc.size_bytes {
            return Err(Error::Buffer(
                "external bytes do not match descriptor size".to_string(),
            ));
        }
        Ok(Self {
            device,
            domain,
            desc,
            external,
            storage: Arc::new(BufferStorage::External {
                bytes: Some(Arc::new(RwLock::new(bytes))),
                _guard: Arc::new(guard),
            }),
        })
    }

    /// Imports an external handle without allocating host storage.
    pub fn from_external(
        device: DeviceKind,
        domain: MemoryDomain,
        desc: BufferDesc,
        external: ExternalHandle,
        guard: ExternalDropGuard,
    ) -> Result<Self> {
        Ok(Self {
            device,
            domain,
            desc,
            external,
            storage: Arc::new(BufferStorage::External {
                bytes: None,
                _guard: Arc::new(guard),
            }),
        })
    }

    pub fn device(&self) -> DeviceKind {
        self.device
    }

    pub fn domain(&self) -> MemoryDomain {
        self.domain
    }

    pub fn memory_type(&self) -> MemoryType {
        match self.domain {
            MemoryDomain::Host => MemoryType::Host,
            MemoryDomain::DmaBuf
            | MemoryDomain::DrmPrime
            | MemoryDomain::VaapiSurface
            | MemoryDomain::CudaDevice
            | MemoryDomain::MppBuffer
            | MemoryDomain::SophonDevice
            | MemoryDomain::Opaque => MemoryType::Device,
        }
    }

    pub fn desc(&self) -> BufferDesc {
        self.desc
    }

    pub fn len(&self) -> usize {
        self.desc.size_bytes
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.storage)
    }

    pub fn read_bytes(&self) -> Vec<u8> {
        self.storage.read_bytes()
    }

    /// Reads bytes only when host storage is already available.
    pub fn try_read_bytes(&self) -> Result<Vec<u8>> {
        if matches!(
            self.storage.as_ref(),
            BufferStorage::External { bytes: None, .. }
        ) {
            return Err(Error::Buffer(
                "external buffer is not host-mapped; call map or stage explicitly".to_string(),
            ));
        }
        Ok(self.storage.read_bytes())
    }

    /// Explicitly maps host-backed storage. External-only buffers require
    /// [`Self::map_with`] because dg-core cannot dereference vendor handles.
    pub fn map(&self) -> Result<Vec<u8>> {
        self.try_read_bytes()
    }

    /// Explicitly stages an external buffer using a caller-provided mapper.
    pub fn map_with(
        &self,
        mapper: impl FnOnce(ExternalHandle, MemoryDomain, usize) -> Result<Vec<u8>>,
    ) -> Result<Vec<u8>> {
        if !matches!(
            self.storage.as_ref(),
            BufferStorage::External { bytes: None, .. }
        ) {
            return self.try_read_bytes();
        }
        let bytes = mapper(self.external, self.domain, self.desc.size_bytes)?;
        if bytes.len() != self.desc.size_bytes {
            return Err(Error::Buffer(
                "mapped external bytes do not match descriptor size".to_string(),
            ));
        }
        Ok(bytes)
    }

    /// Consumes the buffer and returns host bytes, moving them out when possible.
    pub fn into_host_bytes(self) -> Vec<u8> {
        let Self { storage, .. } = self;
        match Arc::try_unwrap(storage) {
            Ok(BufferStorage::Host(bytes)) => match Arc::try_unwrap(bytes) {
                Ok(lock) => match lock.into_inner() {
                    Ok(bytes) => bytes,
                    Err(poisoned) => poisoned.into_inner(),
                },
                Err(bytes) => read_guard(&bytes).clone(),
            },
            Ok(BufferStorage::External {
                bytes: Some(bytes), ..
            }) => match Arc::try_unwrap(bytes) {
                Ok(lock) => match lock.into_inner() {
                    Ok(bytes) => bytes,
                    Err(poisoned) => poisoned.into_inner(),
                },
                Err(bytes) => read_guard(&bytes).clone(),
            },
            Ok(BufferStorage::External { bytes: None, .. }) => Vec::new(),
            Err(storage) => match &*storage {
                BufferStorage::Host(bytes) => read_guard(bytes).clone(),
                BufferStorage::External {
                    bytes: Some(bytes), ..
                } => read_guard(bytes).clone(),
                BufferStorage::External { bytes: None, .. } => Vec::new(),
            },
        }
    }

    pub fn write_from_slice(&self, src: &[u8]) -> Result<()> {
        if src.len() != self.len() {
            return Err(Error::Buffer(
                "source and destination size differ".to_string(),
            ));
        }
        self.storage.write_from_slice(src)
    }

    pub fn copy_into(&self, dst: &mut [u8]) -> Result<()> {
        if dst.len() != self.len() {
            return Err(Error::Buffer(
                "source and destination size differ".to_string(),
            ));
        }
        let bytes = self.try_read_bytes()?;
        dst.copy_from_slice(&bytes);
        Ok(())
    }

    pub fn copy_to(&self, dst: &Buffer) -> Result<()> {
        let bytes = self.try_read_bytes()?;
        dst.write_from_slice(&bytes)
    }

    pub fn external(&self) -> ExternalHandle {
        self.external
    }

    /// Returns whether this buffer has an external handle without host storage.
    pub fn is_external_only(&self) -> bool {
        matches!(
            self.storage.as_ref(),
            BufferStorage::External { bytes: None, .. }
        )
    }
}
