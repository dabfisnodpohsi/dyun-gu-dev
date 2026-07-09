use std::collections::BTreeMap;
use std::sync::Arc;

use dg_core::Tensor;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PacketMeta {
    pub sequence: u64,
    pub stream_id: Option<String>,
    pub tags: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub enum PacketPayload {
    Tensor(Tensor),
    EndOfStream,
}

#[derive(Clone, Debug)]
pub struct Packet {
    pub meta: PacketMeta,
    pub payload: Arc<PacketPayload>,
}

impl Packet {
    pub fn tensor(tensor: Tensor) -> Self {
        Self {
            meta: PacketMeta::default(),
            payload: Arc::new(PacketPayload::Tensor(tensor)),
        }
    }

    pub fn eos() -> Self {
        Self {
            meta: PacketMeta::default(),
            payload: Arc::new(PacketPayload::EndOfStream),
        }
    }

    pub fn is_eos(&self) -> bool {
        matches!(self.payload.as_ref(), PacketPayload::EndOfStream)
    }

    pub fn tensor_ref(&self) -> Option<&Tensor> {
        match self.payload.as_ref() {
            PacketPayload::Tensor(tensor) => Some(tensor),
            PacketPayload::EndOfStream => None,
        }
    }

    pub fn into_tensor(self) -> Option<Tensor> {
        match Arc::try_unwrap(self.payload) {
            Ok(PacketPayload::Tensor(tensor)) => Some(tensor),
            Ok(PacketPayload::EndOfStream) | Err(_) => None,
        }
    }
}
