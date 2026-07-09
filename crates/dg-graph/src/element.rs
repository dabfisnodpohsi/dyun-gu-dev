use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use dg_core::{DataType, Tensor};

use crate::error::{Error, Result};
use crate::packet::Packet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortSchema {
    pub name: &'static str,
    pub dtype: Option<DataType>,
}

#[derive(Clone, Debug, Default)]
pub enum ElementHandle {
    #[default]
    None,
    Sink(Arc<std::sync::Mutex<Vec<Tensor>>>),
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
    pub inputs: HashMap<String, Receiver<Packet>>,
    pub outputs: HashMap<String, Vec<SyncSender<Packet>>>,
    pub stop: Arc<AtomicBool>,
    pub send_backoff: Duration,
}

impl ElementIo {
    pub fn recv(&self, port: &str) -> Result<Option<Packet>> {
        let receiver = self.inputs.get(port).ok_or_else(|| Error::UnknownPort {
            node: self.name.clone(),
            port: port.to_string(),
        })?;
        match receiver.recv_timeout(self.send_backoff) {
            Ok(packet) => Ok(Some(packet)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(Error::Runtime(format!(
                "receive failed on {port}: disconnected"
            ))),
        }
    }

    pub fn send(&self, port: &str, packet: Packet) -> Result<()> {
        let senders = self.outputs.get(port).ok_or_else(|| Error::UnknownPort {
            node: self.name.clone(),
            port: port.to_string(),
        })?;
        for sender in senders {
            loop {
                if self.stop.load(Ordering::Relaxed) {
                    return Err(Error::NotRunning);
                }
                match sender.try_send(packet.clone()) {
                    Ok(()) => break,
                    Err(TrySendError::Full(_)) => thread::sleep(self.send_backoff),
                    Err(TrySendError::Disconnected(_)) => {
                        return Err(Error::Runtime(format!("downstream disconnected on {port}")))
                    }
                }
            }
        }
        Ok(())
    }

    pub fn broadcast_eos(&self) -> Result<()> {
        let packet = Packet::eos();
        for port in self.outputs.keys() {
            self.send(port, packet.clone())?;
        }
        Ok(())
    }
}
