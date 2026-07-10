//! Bounded/unbounded typed packet queues connecting graph elements.

use std::sync::mpsc::{channel, sync_channel, Receiver, RecvTimeoutError, Sender, TrySendError};
use std::time::Duration;

use crate::packet::Packet;

/// Default per-edge queue capacity, matching the sophon-stream default.
pub const DEFAULT_QUEUE_CAPACITY: usize = 20;

/// A typed queue carrying packets between two elements.
///
/// Bounded pipes apply backpressure when full; unbounded pipes are used by
/// dataflow-style execution modes where producers run to completion before
/// their consumers start.
pub struct DataPipe {
    sender: PipeSender,
    receiver: PipeReceiver,
}

impl DataPipe {
    pub fn bounded(capacity: usize) -> Self {
        let (sender, receiver) = sync_channel(capacity);
        Self {
            sender: PipeSender::Bounded(sender),
            receiver: PipeReceiver { receiver },
        }
    }

    pub fn unbounded() -> Self {
        let (sender, receiver) = channel();
        Self {
            sender: PipeSender::Unbounded(sender),
            receiver: PipeReceiver { receiver },
        }
    }

    pub fn split(self) -> (PipeSender, PipeReceiver) {
        (self.sender, self.receiver)
    }
}

/// Sending half of a [`DataPipe`].
#[derive(Clone)]
pub enum PipeSender {
    Bounded(std::sync::mpsc::SyncSender<Packet>),
    Unbounded(Sender<Packet>),
}

impl PipeSender {
    pub fn try_send(&self, packet: Packet) -> std::result::Result<(), TrySendError<Packet>> {
        match self {
            Self::Bounded(sender) => sender.try_send(packet),
            Self::Unbounded(sender) => sender
                .send(packet)
                .map_err(|err| TrySendError::Disconnected(err.0)),
        }
    }
}

/// Receiving half of a [`DataPipe`].
pub struct PipeReceiver {
    receiver: Receiver<Packet>,
}

impl PipeReceiver {
    pub fn recv_timeout(&self, timeout: Duration) -> std::result::Result<Packet, RecvTimeoutError> {
        self.receiver.recv_timeout(timeout)
    }
}
