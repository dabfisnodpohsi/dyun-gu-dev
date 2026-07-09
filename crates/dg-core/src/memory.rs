use std::sync::{Arc, Mutex};

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

impl Clone for ExternalDropGuard {
    fn clone(&self) -> Self {
        Self {
            callback: Arc::clone(&self.callback),
        }
    }
}

impl core::fmt::Debug for ExternalDropGuard {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExternalDropGuard").finish_non_exhaustive()
    }
}
