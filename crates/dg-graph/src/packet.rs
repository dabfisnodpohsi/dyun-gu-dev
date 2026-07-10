use std::collections::BTreeMap;
use std::sync::Arc;

use dg_core::{Detection, Tensor};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PacketMeta {
    pub sequence: u64,
    pub stream_id: Option<String>,
    pub tags: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub enum PacketPayload {
    Tensor(Tensor),
    Detections(Arc<Vec<Detection>>),
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

    pub fn detections(detections: Vec<Detection>) -> Self {
        Self {
            meta: PacketMeta::default(),
            payload: Arc::new(PacketPayload::Detections(Arc::new(detections))),
        }
    }

    pub fn is_eos(&self) -> bool {
        matches!(self.payload.as_ref(), PacketPayload::EndOfStream)
    }

    pub fn tensor_ref(&self) -> Option<&Tensor> {
        match self.payload.as_ref() {
            PacketPayload::Tensor(tensor) => Some(tensor),
            PacketPayload::Detections(_) | PacketPayload::EndOfStream => None,
        }
    }

    pub fn detections_ref(&self) -> Option<&[Detection]> {
        match self.payload.as_ref() {
            PacketPayload::Detections(detections) => Some(detections.as_slice()),
            PacketPayload::Tensor(_) | PacketPayload::EndOfStream => None,
        }
    }

    pub fn into_detections(self) -> Option<Vec<Detection>> {
        match Arc::try_unwrap(self.payload) {
            Ok(PacketPayload::Detections(detections)) => Some(Arc::unwrap_or_clone(detections)),
            Ok(PacketPayload::Tensor(_) | PacketPayload::EndOfStream) => None,
            Err(payload) => match payload.as_ref() {
                PacketPayload::Detections(detections) => Some(detections.as_ref().clone()),
                PacketPayload::Tensor(_) | PacketPayload::EndOfStream => None,
            },
        }
    }

    pub fn with_meta(mut self, meta: PacketMeta) -> Self {
        self.meta = meta;
        self
    }

    pub fn into_tensor(self) -> Option<Tensor> {
        match Arc::try_unwrap(self.payload) {
            Ok(PacketPayload::Tensor(tensor)) => Some(tensor),
            Ok(PacketPayload::Detections(_) | PacketPayload::EndOfStream) => None,
            Err(payload) => match payload.as_ref() {
                PacketPayload::Tensor(tensor) => Some(tensor.clone()),
                PacketPayload::Detections(_) | PacketPayload::EndOfStream => None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use dg_core::{
        BBox, CpuDevice, DataFormat, DataType, Detection, DeviceKind, Shape, Tensor, TensorDesc,
    };

    use super::Packet;

    fn test_tensor() -> Tensor {
        let device = CpuDevice::new();
        let desc = TensorDesc::new(
            Shape::new([1, 4]),
            DataType::U8,
            DataFormat::NC,
            DeviceKind::Cpu,
        );
        let tensor = Tensor::allocate(&device, desc).expect("allocate test tensor");
        tensor
            .buffer()
            .write_from_slice(&[1, 2, 3, 4])
            .expect("write test tensor");
        tensor
    }

    #[test]
    fn into_tensor_preserves_shared_tensor_payload() {
        let packet = Packet::tensor(test_tensor());
        let cloned_packet = packet.clone();

        assert_eq!(
            cloned_packet
                .into_tensor()
                .expect("shared tensor payload")
                .buffer()
                .read_bytes(),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn into_detections_preserves_shared_payload() {
        let packet =
            Packet::detections(vec![Detection::new(BBox::new(1.0, 2.0, 3.0, 4.0), 0.9, 7)]);
        let cloned_packet = packet.clone();
        let detections = cloned_packet
            .into_detections()
            .expect("shared detections payload");
        assert_eq!(detections[0].class_id, 7);
    }
}
