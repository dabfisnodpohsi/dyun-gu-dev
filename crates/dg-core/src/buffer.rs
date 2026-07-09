use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::{DeviceKind, Error, MemoryType, Result};

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
    External(Arc<[u8]>),
}

impl BufferStorage {
    fn len(&self) -> usize {
        match self {
            Self::Host(bytes) => read_guard(bytes).len(),
            Self::External(bytes) => bytes.len(),
        }
    }

    fn read_bytes(&self) -> Vec<u8> {
        match self {
            Self::Host(bytes) => read_guard(bytes).clone(),
            Self::External(bytes) => bytes.to_vec(),
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
            Self::External(_) => Err(Error::Buffer("buffer is immutable".to_string())),
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
    memory_type: MemoryType,
    desc: BufferDesc,
    storage: Arc<BufferStorage>,
}

impl Buffer {
    pub(crate) fn new_host(device: DeviceKind, desc: BufferDesc) -> Self {
        Self {
            device,
            memory_type: MemoryType::Host,
            desc,
            storage: Arc::new(BufferStorage::Host(Arc::new(RwLock::new(vec![
                0;
                desc.size_bytes
            ])))),
        }
    }

    pub fn from_external(device: DeviceKind, desc: BufferDesc, bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() != desc.size_bytes {
            return Err(Error::Buffer(
                "external bytes do not match descriptor size".to_string(),
            ));
        }
        Ok(Self {
            device,
            memory_type: MemoryType::Host,
            desc,
            storage: Arc::new(BufferStorage::External(Arc::<[u8]>::from(
                bytes.into_boxed_slice(),
            ))),
        })
    }

    pub fn device(&self) -> DeviceKind {
        self.device
    }

    pub fn memory_type(&self) -> MemoryType {
        self.memory_type
    }

    pub fn desc(&self) -> BufferDesc {
        self.desc
    }

    pub fn len(&self) -> usize {
        self.storage.len()
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
        dst.copy_from_slice(&self.storage.read_bytes());
        Ok(())
    }

    pub fn copy_to(&self, dst: &Buffer) -> Result<()> {
        let bytes = self.read_bytes();
        dst.write_from_slice(&bytes)
    }
}
