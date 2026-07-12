use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{RecvTimeoutError, TrySendError};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use dg_core::{Classification, DataType, Detection, FaceDetection, OcrText, Tensor, Track};

use crate::error::{Error, Result};
use crate::metrics::ElementMetrics;
use crate::packet::Packet;
use crate::pipe::{PipeReceiver, PipeSender};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortSchema {
    pub name: &'static str,
    pub dtype: Option<DataType>,
    /// For input ports, whether exactly one incoming connection is required.
    /// This flag is ignored for output ports.
    pub required: bool,
}

#[derive(Clone, Debug, Default)]
pub enum ElementHandle {
    #[default]
    None,
    Input(Arc<Mutex<VecDeque<Tensor>>>),
    Sink(Arc<std::sync::Mutex<SinkCollector>>),
}

#[derive(Clone, Debug, Default)]
pub struct SinkCollector {
    pub tensors: Vec<Tensor>,
    pub detections: Vec<Vec<Detection>>,
    pub classifications: Vec<Vec<Classification>>,
    pub faces: Vec<Vec<FaceDetection>>,
    pub tracks: Vec<Vec<Track>>,
    pub ocr: Vec<Vec<OcrText>>,
}

pub struct CreatedElement {
    pub element: Box<dyn Element>,
    pub handle: ElementHandle,
}

pub trait Element: Send {
    fn run(self: Box<Self>, io: ElementIo) -> Result<()>;
}

pub struct ElementIo {
    pub name: String,
    pub inputs: HashMap<String, Arc<Mutex<PipeReceiver>>>,
    pub outputs: HashMap<String, Vec<PipeSender>>,
    pub stop: Arc<AtomicBool>,
    pub send_backoff: Duration,
    pub(crate) eos: Arc<Mutex<EosState>>,
    pub(crate) metrics: Arc<ElementMetrics>,
    pub(crate) packet_starts: RefCell<VecDeque<Instant>>,
}

impl ElementIo {
    pub fn recv(&self, port: &str) -> Result<Option<Packet>> {
        let receiver = self.inputs.get(port).ok_or_else(|| Error::UnknownPort {
            node: self.name.clone(),
            port: port.to_string(),
        })?;
        let receiver = receiver
            .lock()
            .map_err(|_| Error::Runtime(format!("receive lock poisoned on {port}")))?;
        let result = receiver.recv_timeout(self.send_backoff);
        self.metrics.record_queue_depth(receiver.depth());
        drop(receiver);
        match result {
            Ok(packet) => {
                if packet.is_eos() {
                    self.eos
                        .lock()
                        .map_err(|_| Error::Runtime("EOS state lock poisoned".to_string()))?
                        .seen = true;
                } else {
                    self.metrics.record_received();
                    self.packet_starts.borrow_mut().push_back(Instant::now());
                }
                Ok(Some(packet))
            }
            Err(RecvTimeoutError::Timeout) => {
                let seen = self
                    .eos
                    .lock()
                    .map_err(|_| Error::Runtime("EOS state lock poisoned".to_string()))?
                    .seen;
                if seen {
                    Ok(Some(Packet::eos()))
                } else {
                    Ok(None)
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                let seen = self
                    .eos
                    .lock()
                    .map_err(|_| Error::Runtime("EOS state lock poisoned".to_string()))?
                    .seen;
                if seen {
                    Ok(Some(Packet::eos()))
                } else {
                    Err(Error::Runtime(format!(
                        "receive failed on {port}: disconnected"
                    )))
                }
            }
        }
    }

    pub fn send(&self, port: &str, packet: Packet) -> Result<()> {
        let senders = self.outputs.get(port).ok_or_else(|| Error::UnknownPort {
            node: self.name.clone(),
            port: port.to_string(),
        })?;
        let is_eos = packet.is_eos();
        let is_source = self.inputs.is_empty();
        for sender in senders {
            loop {
                if self.stop.load(Ordering::Relaxed) {
                    return Err(Error::NotRunning);
                }
                match sender.try_send(packet.clone()) {
                    Ok(()) => {
                        self.metrics.record_queue_depth(sender.depth());
                        break;
                    }
                    Err(TrySendError::Full(_)) => {
                        self.metrics.record_backpressure();
                        self.metrics.record_queue_depth(sender.depth());
                        thread::sleep(self.send_backoff);
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        self.metrics.record_drop();
                        return Err(Error::Runtime(format!("downstream disconnected on {port}")));
                    }
                }
            }
        }
        if !is_eos {
            self.metrics.record_sent();
            if is_source {
                self.metrics.record_source_packet();
            } else {
                self.complete_packet()?;
            }
        }
        Ok(())
    }

    pub fn finish_packet(&self) -> Result<()> {
        self.complete_packet()
    }

    pub fn drop_packet(&self) -> Result<()> {
        self.metrics.record_drop();
        self.complete_packet()
    }

    fn complete_packet(&self) -> Result<()> {
        if let Some(started) = self.packet_starts.borrow_mut().pop_front() {
            self.metrics.record_latency(started.elapsed());
        }
        Ok(())
    }

    pub fn broadcast_eos(&self) -> Result<()> {
        let should_broadcast = {
            let mut eos = self
                .eos
                .lock()
                .map_err(|_| Error::Runtime("EOS state lock poisoned".to_string()))?;
            eos.broadcasts += 1;
            eos.broadcasts == eos.instances
        };
        if !should_broadcast {
            return Ok(());
        }
        let packet = Packet::eos();
        for port in self.outputs.keys() {
            self.send(port, packet.clone())?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct EosState {
    pub seen: bool,
    pub broadcasts: usize,
    pub instances: usize,
}
