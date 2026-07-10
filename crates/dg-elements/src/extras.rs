use std::collections::VecDeque;

use dg_core::{
    BBox, Classification, DataFormat, DataType, Detection, DeviceKind, FaceDetection, OcrText,
    Point, Shape, Tensor, TensorDesc, Track, TrackState,
};
use dg_graph::{
    CreatedElement, Element, ElementHandle, ElementIo, Error, NodeSpec, Packet, PortSchema, Result,
};

use crate::math::{iou, resize_letterbox, softmax, top_k};

const ANY_INPUT: [PortSchema; 1] = [PortSchema {
    name: "in",
    dtype: None,
}];
const TENSOR_INPUT: [PortSchema; 1] = [PortSchema {
    name: "in",
    dtype: Some(DataType::F32),
}];
const TENSOR_OUTPUT: [PortSchema; 1] = [PortSchema {
    name: "out",
    dtype: Some(DataType::F32),
}];
const RESULT_OUTPUT: [PortSchema; 1] = [PortSchema {
    name: "out",
    dtype: None,
}];

inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "resnet_preprocess",
        input_ports: &ANY_INPUT,
        output_ports: &TENSOR_OUTPUT,
        create: create_resnet_preprocess,
    }
}
inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "resnet_postprocess",
        input_ports: &TENSOR_INPUT,
        output_ports: &RESULT_OUTPUT,
        create: create_resnet_postprocess,
    }
}
inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "retinaface",
        input_ports: &TENSOR_INPUT,
        output_ports: &RESULT_OUTPUT,
        create: create_retinaface,
    }
}
inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "bytetrack",
        input_ports: &ANY_INPUT,
        output_ports: &RESULT_OUTPUT,
        create: create_bytetrack,
    }
}
inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "ppocr_det",
        input_ports: &TENSOR_INPUT,
        output_ports: &RESULT_OUTPUT,
        create: create_ppocr_det,
    }
}
inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "ppocr_rec",
        input_ports: &TENSOR_INPUT,
        output_ports: &RESULT_OUTPUT,
        create: create_ppocr_rec,
    }
}

struct ResnetPreprocess {
    width: usize,
    height: usize,
    mean: [f32; 3],
    std: [f32; 3],
}

struct ResnetPostprocess {
    top_k: usize,
    labels: Vec<String>,
}

struct Retinaface {
    width: usize,
    height: usize,
    score_threshold: f32,
    nms_threshold: f32,
    anchors: Vec<BBox>,
}

struct ByteTrack {
    next_id: u64,
    max_lost: u32,
    match_threshold: f32,
    tracks: Vec<TrackStateInner>,
}

struct TrackStateInner {
    track_id: u64,
    detection: Detection,
    lost: u32,
}

struct PpocrDet {
    threshold: f32,
}

struct PpocrRec {
    alphabet: Vec<char>,
    blank: usize,
}

impl Element for ResnetPreprocess {
    fn run(self: Box<Self>, io: ElementIo) -> Result<()> {
        loop {
            let packet = next_packet(&io)?;
            if packet.is_eos() {
                io.broadcast_eos()?;
                return Ok(());
            }
            let tensor = resnet_preprocess_tensor(
                packet.tensor_ref().ok_or_else(|| {
                    Error::Runtime("resnet preprocess expects a tensor".to_string())
                })?,
                self.width,
                self.height,
                self.mean,
                self.std,
            )?;
            io.send("out", Packet::tensor(tensor).with_meta(packet.meta))?;
        }
    }
}

impl Element for ResnetPostprocess {
    fn run(self: Box<Self>, io: ElementIo) -> Result<()> {
        loop {
            let packet = next_packet(&io)?;
            if packet.is_eos() {
                io.broadcast_eos()?;
                return Ok(());
            }
            let values = f32_values(packet.tensor_ref().ok_or_else(|| {
                Error::Runtime("resnet postprocess expects a tensor".to_string())
            })?)?;
            let probabilities = softmax(&values);
            let results = top_k(&probabilities, self.top_k)
                .into_iter()
                .map(|(index, score)| {
                    Ok(Classification {
                        class_id: u32::try_from(index)
                            .map_err(|_| Error::Runtime("class id is out of range".to_string()))?,
                        score,
                        label: self.labels.get(index).cloned(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            io.send(
                "out",
                Packet::classifications(results).with_meta(packet.meta),
            )?;
        }
    }
}

impl Element for Retinaface {
    fn run(self: Box<Self>, io: ElementIo) -> Result<()> {
        loop {
            let packet = next_packet(&io)?;
            if packet.is_eos() {
                io.broadcast_eos()?;
                return Ok(());
            }
            let values = f32_values(
                packet
                    .tensor_ref()
                    .ok_or_else(|| Error::Runtime("retinaface expects a tensor".to_string()))?,
            )?;
            let faces = decode_retinaface(
                &values,
                &self.anchors,
                self.width,
                self.height,
                self.score_threshold,
                self.nms_threshold,
            )?;
            io.send("out", Packet::faces(faces).with_meta(packet.meta))?;
        }
    }
}

impl Element for ByteTrack {
    fn run(mut self: Box<Self>, io: ElementIo) -> Result<()> {
        loop {
            let packet = next_packet(&io)?;
            if packet.is_eos() {
                io.broadcast_eos()?;
                return Ok(());
            }
            let detections = packet.detections_ref().ok_or_else(|| {
                Error::Runtime("bytetrack expects detections payload".to_string())
            })?;
            let results = self.update(detections);
            io.send("out", Packet::tracks(results).with_meta(packet.meta))?;
        }
    }
}

impl Element for PpocrDet {
    fn run(self: Box<Self>, io: ElementIo) -> Result<()> {
        loop {
            let packet = next_packet(&io)?;
            if packet.is_eos() {
                io.broadcast_eos()?;
                return Ok(());
            }
            let tensor = packet
                .tensor_ref()
                .ok_or_else(|| Error::Runtime("ppocr det expects a tensor".to_string()))?;
            let results = detect_text_regions(tensor, self.threshold)?;
            io.send("out", Packet::ocr(results).with_meta(packet.meta))?;
        }
    }
}

impl Element for PpocrRec {
    fn run(self: Box<Self>, io: ElementIo) -> Result<()> {
        loop {
            let packet = next_packet(&io)?;
            if packet.is_eos() {
                io.broadcast_eos()?;
                return Ok(());
            }
            let logits = f32_values(
                packet
                    .tensor_ref()
                    .ok_or_else(|| Error::Runtime("ppocr rec expects a tensor".to_string()))?,
            )?;
            let class_count = self
                .alphabet
                .len()
                .checked_add(1)
                .ok_or_else(|| Error::Runtime("ocr alphabet size overflow".to_string()))?;
            if class_count == 0 || logits.len() % class_count != 0 {
                return Err(Error::Runtime(
                    "ocr logits do not match alphabet size".to_string(),
                ));
            }
            let rows = logits
                .chunks_exact(class_count)
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();
            let text = ctc_greedy_decode(&rows, &self.alphabet, self.blank)?;
            io.send(
                "out",
                Packet::ocr(vec![OcrText {
                    text,
                    score: 1.0,
                    bbox: None,
                }])
                .with_meta(packet.meta),
            )?;
        }
    }
}

impl ByteTrack {
    fn update(&mut self, detections: &[Detection]) -> Vec<Track> {
        for track in &mut self.tracks {
            track.lost = track.lost.saturating_add(1);
        }
        let mut matched = vec![false; self.tracks.len()];
        let mut output = Vec::new();
        for detection in detections {
            let best = self
                .tracks
                .iter()
                .enumerate()
                .filter(|(index, track)| {
                    !matched[*index]
                        && track.detection.class_id == detection.class_id
                        && iou(track.detection.bbox, detection.bbox) >= self.match_threshold
                })
                .max_by(|left, right| {
                    iou(left.1.detection.bbox, detection.bbox)
                        .total_cmp(&iou(right.1.detection.bbox, detection.bbox))
                })
                .map(|(index, _)| index);
            let (track_id, state) = if let Some(index) = best {
                matched[index] = true;
                let track = &mut self.tracks[index];
                track.detection = detection.clone();
                track.lost = 0;
                (track.track_id, TrackState::Tracked)
            } else {
                let track_id = self.next_id;
                self.next_id = self.next_id.saturating_add(1);
                self.tracks.push(TrackStateInner {
                    track_id,
                    detection: detection.clone(),
                    lost: 0,
                });
                matched.push(true);
                (track_id, TrackState::New)
            };
            output.push(Track {
                track_id,
                detection: detection.clone(),
                state,
            });
        }
        self.tracks.retain(|track| track.lost <= self.max_lost);
        output
    }
}

pub fn generate_anchors(width: usize, height: usize, stride: usize, sizes: &[f32]) -> Vec<BBox> {
    if stride == 0 {
        return Vec::new();
    }
    let mut anchors = Vec::new();
    for y in (0..height).step_by(stride) {
        for x in (0..width).step_by(stride) {
            let center_x = dimension_or_zero(x) / dimension_or_one(width);
            let center_y = dimension_or_zero(y) / dimension_or_one(height);
            for size in sizes {
                anchors.push(BBox::new(center_x, center_y, *size, *size));
            }
        }
    }
    anchors
}

fn decode_retinaface(
    values: &[f32],
    anchors: &[BBox],
    width: usize,
    height: usize,
    score_threshold: f32,
    nms_threshold: f32,
) -> Result<Vec<FaceDetection>> {
    const ATTRIBUTES: usize = 15;
    if values.len().checked_rem(ATTRIBUTES) != Some(0) {
        return Err(Error::Runtime(
            "retinaface output must have 15 values per anchor".to_string(),
        ));
    }
    let width_f = usize_f32(width)?;
    let height_f = usize_f32(height)?;
    let mut candidates = Vec::new();
    for (index, row) in values.chunks_exact(ATTRIBUTES).enumerate() {
        let anchor = match anchors.get(index).copied() {
            Some(anchor) => anchor,
            None => BBox::new(0.5, 0.5, 1.0 / width_f, 1.0 / height_f),
        };
        let score = crate::math::sigmoid(row[4]);
        if score < score_threshold {
            continue;
        }
        let bbox = BBox::new(
            (anchor.x + row[0] * 0.1 * anchor.w).clamp(0.0, 1.0) * width_f,
            (anchor.y + row[1] * 0.1 * anchor.h).clamp(0.0, 1.0) * height_f,
            (anchor.w * row[2].exp()).clamp(0.0, 1.0) * width_f,
            (anchor.h * row[3].exp()).clamp(0.0, 1.0) * height_f,
        );
        let mut landmarks = Vec::with_capacity(5);
        for point in row[5..].chunks_exact(2) {
            landmarks.push(Point {
                x: (anchor.x + point[0] * 0.1 * anchor.w).clamp(0.0, 1.0) * width_f,
                y: (anchor.y + point[1] * 0.1 * anchor.h).clamp(0.0, 1.0) * height_f,
            });
        }
        candidates.push(FaceDetection {
            bbox,
            score,
            landmarks,
        });
    }
    candidates.sort_by(|left, right| right.score.total_cmp(&left.score));
    let mut selected = Vec::new();
    for candidate in candidates {
        if selected
            .iter()
            .all(|existing: &FaceDetection| iou(existing.bbox, candidate.bbox) <= nms_threshold)
        {
            selected.push(candidate);
        }
    }
    Ok(selected)
}

pub fn ctc_greedy_decode(rows: &[Vec<f32>], alphabet: &[char], blank: usize) -> Result<String> {
    let mut output = String::new();
    let mut previous = None;
    for row in rows {
        let index = row
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .ok_or_else(|| Error::Runtime("empty CTC row".to_string()))?;
        if index != blank && Some(index) != previous {
            let character = alphabet
                .get(index)
                .ok_or_else(|| Error::Runtime("CTC class exceeds alphabet".to_string()))?;
            output.push(*character);
        }
        previous = Some(index);
    }
    Ok(output)
}

fn detect_text_regions(tensor: &Tensor, threshold: f32) -> Result<Vec<OcrText>> {
    let dims = tensor.desc().shape().dims();
    let (height, width) = match dims {
        [1, 1, height, width] | [1, height, width] => (*height, *width),
        _ => {
            return Err(Error::Config(
                "ppocr det expects [1,1,H,W] or [1,H,W]".to_string(),
            ))
        }
    };
    let values = f32_values(tensor)?;
    let expected = height
        .checked_mul(width)
        .ok_or_else(|| Error::Runtime("ocr map dimensions overflow".to_string()))?;
    if values.len() != expected {
        return Err(Error::Runtime("ocr map size mismatch".to_string()));
    }
    let mut visited = vec![false; expected];
    let mut output = Vec::new();
    for start in 0..expected {
        if visited[start] || values[start] < threshold {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        visited[start] = true;
        let mut points = Vec::new();
        while let Some(index) = queue.pop_front() {
            points.push(index);
            let y = index / width;
            let x = index % width;
            for (next_x, next_y) in [
                (x.saturating_sub(1), y),
                (x.saturating_add(1), y),
                (x, y.saturating_sub(1)),
                (x, y.saturating_add(1)),
            ] {
                if next_x >= width || next_y >= height {
                    continue;
                }
                let next = next_y * width + next_x;
                if !visited[next] && values[next] >= threshold {
                    visited[next] = true;
                    queue.push_back(next);
                }
            }
        }
        let first = points[0];
        let mut min_x = first % width;
        let mut max_x = min_x;
        let mut min_y = first / width;
        let mut max_y = min_y;
        for index in points.iter().copied().skip(1) {
            min_x = min_x.min(index % width);
            max_x = max_x.max(index % width);
            min_y = min_y.min(index / width);
            max_y = max_y.max(index / width);
        }
        let width_f = usize_f32(width)?;
        let height_f = usize_f32(height)?;
        let score =
            points.iter().map(|index| values[*index]).sum::<f32>() / usize_f32(points.len())?;
        output.push(OcrText {
            text: String::new(),
            score,
            bbox: Some(BBox::new(
                usize_f32(min_x)? / width_f,
                usize_f32(min_y)? / height_f,
                usize_f32(max_x.saturating_sub(min_x).saturating_add(1))? / width_f,
                usize_f32(max_y.saturating_sub(min_y).saturating_add(1))? / height_f,
            )),
        });
    }
    Ok(output)
}

fn resnet_preprocess_tensor(
    input: &Tensor,
    width: usize,
    height: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<Tensor> {
    let dims = input.desc().shape().dims();
    let (channels, source_height, source_width) = match (input.desc().format(), dims) {
        (DataFormat::NCHW, [1, channels, height, width])
        | (DataFormat::NCHW, [channels, height, width]) => (*channels, *height, *width),
        _ => {
            return Err(Error::Config(
                "resnet preprocess expects NCHW rank 3/4 input".to_string(),
            ))
        }
    };
    if channels != 3 {
        return Err(Error::Config("resnet expects three channels".to_string()));
    }
    let values = tensor_values(input)?;
    let mut hwc = vec![0.0; values.len()];
    for channel in 0..channels {
        for y in 0..source_height {
            for x in 0..source_width {
                let source = (channel * source_height + y) * source_width + x;
                let target = (y * source_width + x) * channels + channel;
                hwc[target] = values[source];
            }
        }
    }
    let (resized, _) = resize_letterbox(
        &hwc,
        channels,
        source_width,
        source_height,
        width,
        height,
        0.0,
    )
    .map_err(Error::Config)?;
    let device = dg_core::CpuDevice::new();
    let output = Tensor::allocate(
        &device,
        TensorDesc::new(
            Shape::new([1, 3, height, width]),
            DataType::F32,
            DataFormat::NCHW,
            DeviceKind::Cpu,
        ),
    )?;
    let mut bytes = Vec::new();
    for channel in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let value = resized[(y * width + x) * 3 + channel] / 255.0;
                let normalized = (value - mean[channel]) / std[channel];
                bytes.extend_from_slice(&normalized.to_ne_bytes());
            }
        }
    }
    output.buffer().write_from_slice(&bytes)?;
    Ok(output)
}

fn create_resnet_preprocess(node: &NodeSpec) -> Result<CreatedElement> {
    let params = params_object(node)?;
    let width = read_usize(params, "input_width", 224)?;
    let height = read_usize(params, "input_height", width)?;
    let mean = read_f32_array(params, "mean", [0.485, 0.456, 0.406])?;
    let std = read_f32_array(params, "std", [0.229, 0.224, 0.225])?;
    Ok(CreatedElement {
        element: Box::new(ResnetPreprocess {
            width,
            height,
            mean,
            std,
        }),
        handle: ElementHandle::None,
    })
}

fn create_resnet_postprocess(node: &NodeSpec) -> Result<CreatedElement> {
    let params = params_object(node)?;
    let top_k_value = read_usize(params, "top_k", 5)?;
    let labels = params
        .get("labels")
        .and_then(serde_json::Value::as_array)
        .map_or_else(Vec::new, |values| {
            values
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        });
    Ok(CreatedElement {
        element: Box::new(ResnetPostprocess {
            top_k: top_k_value,
            labels,
        }),
        handle: ElementHandle::None,
    })
}

fn create_retinaface(node: &NodeSpec) -> Result<CreatedElement> {
    let params = params_object(node)?;
    let width = read_usize(params, "image_width", 640)?;
    let height = read_usize(params, "image_height", width)?;
    let stride = read_usize(params, "stride", 16)?;
    let score_threshold = read_f32(params, "confidence_threshold", 0.5)?;
    let nms_threshold = read_f32(params, "nms_threshold", 0.4)?;
    let sizes = read_f32_vec(params, "anchor_sizes")?;
    let sizes = if sizes.is_empty() {
        vec![0.1, 0.2]
    } else {
        sizes
    };
    let anchors = generate_anchors(width, height, stride, &sizes);
    Ok(CreatedElement {
        element: Box::new(Retinaface {
            width,
            height,
            score_threshold,
            nms_threshold,
            anchors,
        }),
        handle: ElementHandle::None,
    })
}

fn create_bytetrack(node: &NodeSpec) -> Result<CreatedElement> {
    let params = params_object(node)?;
    Ok(CreatedElement {
        element: Box::new(ByteTrack {
            next_id: 1,
            max_lost: read_usize(params, "max_lost", 2)?
                .try_into()
                .map_err(|_| Error::Config("max_lost is out of range".to_string()))?,
            match_threshold: read_f32(params, "match_iou", 0.3)?,
            tracks: Vec::new(),
        }),
        handle: ElementHandle::None,
    })
}

fn create_ppocr_det(node: &NodeSpec) -> Result<CreatedElement> {
    let params = params_object(node)?;
    Ok(CreatedElement {
        element: Box::new(PpocrDet {
            threshold: read_f32(params, "threshold", 0.3)?,
        }),
        handle: ElementHandle::None,
    })
}

fn create_ppocr_rec(node: &NodeSpec) -> Result<CreatedElement> {
    let params = params_object(node)?;
    let alphabet: Vec<char> = params
        .get("alphabet")
        .and_then(serde_json::Value::as_str)
        .map_or_else(
            || "0123456789".chars().collect(),
            |value| value.chars().collect(),
        );
    let blank = read_usize(params, "blank_index", alphabet.len())?;
    Ok(CreatedElement {
        element: Box::new(PpocrRec { alphabet, blank }),
        handle: ElementHandle::None,
    })
}

fn next_packet(io: &ElementIo) -> Result<Packet> {
    loop {
        if let Some(packet) = io.recv("in")? {
            return Ok(packet);
        }
        if io.stop.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(Error::NotRunning);
        }
    }
}

fn tensor_values(tensor: &Tensor) -> Result<Vec<f32>> {
    match tensor.desc().dtype() {
        DataType::U8 => Ok(tensor
            .buffer()
            .read_bytes()
            .into_iter()
            .map(f32::from)
            .collect()),
        DataType::F32 => f32_values(tensor),
        dtype => Err(Error::Config(format!(
            "algorithm elements support only u8/f32 tensors, got {dtype:?}"
        ))),
    }
}

fn f32_values(tensor: &Tensor) -> Result<Vec<f32>> {
    let bytes = tensor.buffer().read_bytes();
    let chunks = bytes.chunks_exact(std::mem::size_of::<f32>());
    if !chunks.remainder().is_empty() {
        return Err(Error::Runtime("f32 tensor has partial element".to_string()));
    }
    chunks
        .map(|chunk| {
            let bytes: [u8; 4] = chunk
                .try_into()
                .map_err(|_| Error::Runtime("invalid f32 tensor element".to_string()))?;
            Ok(f32::from_ne_bytes(bytes))
        })
        .collect()
}

fn params_object(node: &NodeSpec) -> Result<&serde_json::Map<String, serde_json::Value>> {
    node.params
        .as_object()
        .ok_or_else(|| Error::Config(format!("node {} params must be an object", node.name)))
}

fn read_usize(
    params: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: usize,
) -> Result<usize> {
    params.get(key).map_or(Ok(default), |value| {
        let value = value
            .as_u64()
            .ok_or_else(|| Error::Config(format!("field {key} must be an integer")))?;
        usize::try_from(value).map_err(|_| Error::Config(format!("field {key} out of range")))
    })
}

fn read_f32(
    params: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: f32,
) -> Result<f32> {
    params.get(key).map_or(Ok(default), |value| {
        let value = value
            .as_f64()
            .ok_or_else(|| Error::Config(format!("field {key} must be a number")))?;
        let narrowed = value
            .to_string()
            .parse::<f32>()
            .map_err(|_| Error::Config(format!("field {key} out of range")))?;
        if narrowed.is_finite() {
            Ok(narrowed)
        } else {
            Err(Error::Config(format!("field {key} must be finite")))
        }
    })
}

fn read_f32_array(
    params: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: [f32; 3],
) -> Result<[f32; 3]> {
    let Some(values) = params.get(key).and_then(serde_json::Value::as_array) else {
        return Ok(default);
    };
    if values.len() != 3 {
        return Err(Error::Config(format!(
            "field {key} must contain three values"
        )));
    }
    let mut output = default;
    for (index, value) in values.iter().enumerate() {
        output[index] = value
            .as_f64()
            .ok_or_else(|| Error::Config(format!("field {key} must contain numbers")))?
            .to_string()
            .parse::<f32>()
            .map_err(|_| Error::Config(format!("field {key} out of range")))?;
    }
    Ok(output)
}

fn read_f32_vec(
    params: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<Vec<f32>> {
    params
        .get(key)
        .and_then(serde_json::Value::as_array)
        .map_or(Ok(Vec::new()), |values| {
            values
                .iter()
                .map(|value| {
                    value
                        .as_f64()
                        .ok_or_else(|| Error::Config(format!("field {key} must contain numbers")))?
                        .to_string()
                        .parse::<f32>()
                        .map_err(|_| Error::Config(format!("field {key} out of range")))
                })
                .collect()
        })
}

fn usize_f32(value: usize) -> Result<f32> {
    value
        .to_string()
        .parse::<f32>()
        .map_err(|_| Error::Runtime("dimension cannot be represented as f32".to_string()))
}

fn dimension_or_zero(value: usize) -> f32 {
    value.to_string().parse::<f32>().map_or(0.0, |value| value)
}

fn dimension_or_one(value: usize) -> f32 {
    match value.to_string().parse::<f32>() {
        Ok(value) if value != 0.0 => value,
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctc_decodes_repeated_symbols_and_blank() {
        let rows = vec![
            vec![0.9, 0.1, 0.0],
            vec![0.8, 0.1, 0.0],
            vec![0.1, 0.9, 0.0],
            vec![0.1, 0.9, 0.0],
        ];
        assert_eq!(
            ctc_greedy_decode(&rows, &['a', 'b'], 2).expect("decode"),
            "ab"
        );
    }

    #[test]
    fn anchor_generation_and_retina_decode_are_bounded() {
        let anchors = generate_anchors(32, 32, 16, &[0.25]);
        let values = vec![0.0; 15];
        let faces = decode_retinaface(&values, &anchors, 32, 32, 0.4, 0.5).expect("decode");
        assert_eq!(faces.len(), 1);
        assert!(faces[0]
            .landmarks
            .iter()
            .all(|point| { (0.0..=32.0).contains(&point.x) && (0.0..=32.0).contains(&point.y) }));
    }

    #[test]
    fn bytetrack_keeps_ids_and_reclaims_expired_tracks() {
        let detection = Detection::new(BBox::new(0.0, 0.0, 10.0, 10.0), 0.9, 0);
        let mut tracker = ByteTrack {
            next_id: 1,
            max_lost: 1,
            match_threshold: 0.3,
            tracks: Vec::new(),
        };
        let first = tracker.update(std::slice::from_ref(&detection));
        let second = tracker.update(std::slice::from_ref(&detection));
        assert_eq!(first[0].track_id, second[0].track_id);
        assert_eq!(second[0].state, TrackState::Tracked);
        assert!(tracker.update(&[]).is_empty());
        assert!(tracker.update(&[]).is_empty());
        let replacement = tracker.update(std::slice::from_ref(&detection));
        assert_ne!(replacement[0].track_id, first[0].track_id);
    }
}
