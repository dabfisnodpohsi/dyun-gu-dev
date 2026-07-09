use crate::Result;

/// Stream kind exposed by the execution layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StreamKind {
    Cpu,
}

/// Event kind exposed by the execution layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventKind {
    Cpu,
}

/// Execution stream abstraction.
pub trait Stream: Send + Sync {
    fn kind(&self) -> StreamKind;
    fn synchronize(&self) -> Result<()>;
}

/// Execution event abstraction.
pub trait Event: Send + Sync {
    fn kind(&self) -> EventKind;
    fn record(&self, stream: &dyn Stream) -> Result<()>;
    fn synchronize(&self) -> Result<()>;
}

/// No-op CPU stream.
#[derive(Clone, Debug, Default)]
pub struct CpuStream;

impl Stream for CpuStream {
    fn kind(&self) -> StreamKind {
        StreamKind::Cpu
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

/// No-op CPU event.
#[derive(Clone, Debug, Default)]
pub struct CpuEvent;

impl Event for CpuEvent {
    fn kind(&self) -> EventKind {
        EventKind::Cpu
    }

    fn record(&self, _stream: &dyn Stream) -> Result<()> {
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
