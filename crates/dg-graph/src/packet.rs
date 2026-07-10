use std::collections::BTreeMap;
use std::sync::Arc;

use dg_core::{Classification, Detection, FaceDetection, OcrText, Tensor, Track};

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
    Classifications(Arc<Vec<Classification>>),
    Faces(Arc<Vec<FaceDetection>>),
    Tracks(Arc<Vec<Track>>),
    Ocr(Arc<Vec<OcrText>>),
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

    pub fn classifications(results: Vec<Classification>) -> Self {
        Self {
            meta: PacketMeta::default(),
            payload: Arc::new(PacketPayload::Classifications(Arc::new(results))),
        }
    }

    pub fn faces(results: Vec<FaceDetection>) -> Self {
        Self {
            meta: PacketMeta::default(),
            payload: Arc::new(PacketPayload::Faces(Arc::new(results))),
        }
    }

    pub fn tracks(results: Vec<Track>) -> Self {
        Self {
            meta: PacketMeta::default(),
            payload: Arc::new(PacketPayload::Tracks(Arc::new(results))),
        }
    }

    pub fn ocr(results: Vec<OcrText>) -> Self {
        Self {
            meta: PacketMeta::default(),
            payload: Arc::new(PacketPayload::Ocr(Arc::new(results))),
        }
    }

    pub fn is_eos(&self) -> bool {
        matches!(self.payload.as_ref(), PacketPayload::EndOfStream)
    }

    pub fn tensor_ref(&self) -> Option<&Tensor> {
        match self.payload.as_ref() {
            PacketPayload::Tensor(tensor) => Some(tensor),
            PacketPayload::Detections(_)
            | PacketPayload::Classifications(_)
            | PacketPayload::Faces(_)
            | PacketPayload::Tracks(_)
            | PacketPayload::Ocr(_)
            | PacketPayload::EndOfStream => None,
        }
    }

    pub fn detections_ref(&self) -> Option<&[Detection]> {
        match self.payload.as_ref() {
            PacketPayload::Detections(detections) => Some(detections.as_slice()),
            PacketPayload::Tensor(_)
            | PacketPayload::Classifications(_)
            | PacketPayload::Faces(_)
            | PacketPayload::Tracks(_)
            | PacketPayload::Ocr(_)
            | PacketPayload::EndOfStream => None,
        }
    }

    pub fn into_detections(self) -> Option<Vec<Detection>> {
        match Arc::try_unwrap(self.payload) {
            Ok(PacketPayload::Detections(detections)) => Some(Arc::unwrap_or_clone(detections)),
            Ok(
                PacketPayload::Tensor(_)
                | PacketPayload::Classifications(_)
                | PacketPayload::Faces(_)
                | PacketPayload::Tracks(_)
                | PacketPayload::Ocr(_)
                | PacketPayload::EndOfStream,
            ) => None,
            Err(payload) => match payload.as_ref() {
                PacketPayload::Detections(detections) => Some(detections.as_ref().clone()),
                PacketPayload::Tensor(_)
                | PacketPayload::Classifications(_)
                | PacketPayload::Faces(_)
                | PacketPayload::Tracks(_)
                | PacketPayload::Ocr(_)
                | PacketPayload::EndOfStream => None,
            },
        }
    }

    pub fn classifications_ref(&self) -> Option<&[Classification]> {
        match self.payload.as_ref() {
            PacketPayload::Classifications(results) => Some(results.as_slice()),
            _ => None,
        }
    }

    pub fn into_classifications(self) -> Option<Vec<Classification>> {
        match Arc::try_unwrap(self.payload) {
            Ok(PacketPayload::Classifications(results)) => Some(Arc::unwrap_or_clone(results)),
            Err(payload) => match payload.as_ref() {
                PacketPayload::Classifications(results) => Some(results.as_ref().clone()),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn faces_ref(&self) -> Option<&[FaceDetection]> {
        match self.payload.as_ref() {
            PacketPayload::Faces(results) => Some(results.as_slice()),
            _ => None,
        }
    }

    pub fn into_faces(self) -> Option<Vec<FaceDetection>> {
        match Arc::try_unwrap(self.payload) {
            Ok(PacketPayload::Faces(results)) => Some(Arc::unwrap_or_clone(results)),
            Err(payload) => match payload.as_ref() {
                PacketPayload::Faces(results) => Some(results.as_ref().clone()),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn tracks_ref(&self) -> Option<&[Track]> {
        match self.payload.as_ref() {
            PacketPayload::Tracks(results) => Some(results.as_slice()),
            _ => None,
        }
    }

    pub fn into_tracks(self) -> Option<Vec<Track>> {
        match Arc::try_unwrap(self.payload) {
            Ok(PacketPayload::Tracks(results)) => Some(Arc::unwrap_or_clone(results)),
            Err(payload) => match payload.as_ref() {
                PacketPayload::Tracks(results) => Some(results.as_ref().clone()),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn ocr_ref(&self) -> Option<&[OcrText]> {
        match self.payload.as_ref() {
            PacketPayload::Ocr(results) => Some(results.as_slice()),
            _ => None,
        }
    }

    pub fn into_ocr(self) -> Option<Vec<OcrText>> {
        match Arc::try_unwrap(self.payload) {
            Ok(PacketPayload::Ocr(results)) => Some(Arc::unwrap_or_clone(results)),
            Err(payload) => match payload.as_ref() {
                PacketPayload::Ocr(results) => Some(results.as_ref().clone()),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn with_meta(mut self, meta: PacketMeta) -> Self {
        self.meta = meta;
        self
    }

    pub fn into_tensor(self) -> Option<Tensor> {
        match Arc::try_unwrap(self.payload) {
            Ok(PacketPayload::Tensor(tensor)) => Some(tensor),
            Ok(
                PacketPayload::Detections(_)
                | PacketPayload::Classifications(_)
                | PacketPayload::Faces(_)
                | PacketPayload::Tracks(_)
                | PacketPayload::Ocr(_)
                | PacketPayload::EndOfStream,
            ) => None,
            Err(payload) => match payload.as_ref() {
                PacketPayload::Tensor(tensor) => Some(tensor.clone()),
                PacketPayload::Detections(_)
                | PacketPayload::Classifications(_)
                | PacketPayload::Faces(_)
                | PacketPayload::Tracks(_)
                | PacketPayload::Ocr(_)
                | PacketPayload::EndOfStream => None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use dg_core::{
        BBox, Classification, CpuDevice, DataFormat, DataType, Detection, DeviceKind,
        FaceDetection, OcrText, Point, Shape, Tensor, TensorDesc, Track, TrackState,
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

    #[test]
    fn structured_results_preserve_shared_payloads() {
        let classification = Packet::classifications(vec![Classification {
            class_id: 2,
            score: 0.8,
            label: None,
        }]);
        assert_eq!(
            classification
                .clone()
                .into_classifications()
                .expect("classification")[0]
                .class_id,
            2
        );

        let face = Packet::faces(vec![FaceDetection {
            bbox: BBox::new(0.0, 0.0, 1.0, 1.0),
            score: 0.9,
            landmarks: vec![Point { x: 0.5, y: 0.5 }],
        }]);
        assert_eq!(
            face.clone().into_faces().expect("face")[0].landmarks.len(),
            1
        );

        let track = Packet::tracks(vec![Track {
            track_id: 4,
            detection: Detection::new(BBox::new(0.0, 0.0, 1.0, 1.0), 0.9, 0),
            state: TrackState::Tracked,
        }]);
        assert_eq!(track.clone().into_tracks().expect("track")[0].track_id, 4);

        let ocr = Packet::ocr(vec![OcrText {
            text: "text".to_string(),
            score: 0.9,
            bbox: None,
        }]);
        assert_eq!(ocr.into_ocr().expect("ocr")[0].text, "text");
    }
}
