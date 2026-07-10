use dg_core::{DataFormat, DataType, DeviceKind, Shape, Tensor, TensorDesc};
use dg_graph::{
    CreatedElement, Element, ElementHandle, ElementIo, Error, NodeSpec, Packet, ParamField,
    ParamType, PortSchema, Result,
};

use crate::math::{nms, resize_letterbox, sigmoid, Letterbox};

const PRE_INPUT: [PortSchema; 1] = [PortSchema {
    name: "in",
    dtype: None,
}];
const PRE_OUTPUT: [PortSchema; 1] = [PortSchema {
    name: "out",
    dtype: Some(DataType::F32),
}];
const POST_INPUT: [PortSchema; 1] = [PortSchema {
    name: "in",
    dtype: Some(DataType::F32),
}];
const POST_OUTPUT: [PortSchema; 1] = [PortSchema {
    name: "out",
    dtype: None,
}];
const PREPROCESS_FIELDS: &[&str] = &["input_width", "input_height"];
const POSTPROCESS_FIELDS: &[&str] = &[
    "input_width",
    "input_height",
    "class_count",
    "confidence_threshold",
    "nms_threshold",
];
const PREPROCESS_PARAMS: &[ParamField] = &[
    ParamField {
        name: "input_width",
        ty: ParamType::Uint,
        required: false,
    },
    ParamField {
        name: "input_height",
        ty: ParamType::Uint,
        required: false,
    },
];
const POSTPROCESS_PARAMS: &[ParamField] = &[
    ParamField {
        name: "input_width",
        ty: ParamType::Uint,
        required: false,
    },
    ParamField {
        name: "input_height",
        ty: ParamType::Uint,
        required: false,
    },
    ParamField {
        name: "class_count",
        ty: ParamType::Uint,
        required: false,
    },
    ParamField {
        name: "confidence_threshold",
        ty: ParamType::Float,
        required: false,
    },
    ParamField {
        name: "nms_threshold",
        ty: ParamType::Float,
        required: false,
    },
];

inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "yolo_preprocess",
        input_ports: &PRE_INPUT,
        output_ports: &PRE_OUTPUT,
        params: PREPROCESS_PARAMS,
        validate: Some(validate_preprocess),
        create: create_preprocess,
    }
}

inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "yolo_postprocess",
        input_ports: &POST_INPUT,
        output_ports: &POST_OUTPUT,
        params: POSTPROCESS_PARAMS,
        validate: Some(validate_postprocess),
        create: create_postprocess,
    }
}

struct Preprocess {
    target_width: usize,
    target_height: usize,
}

struct Postprocess {
    target_width: f32,
    target_height: f32,
    class_count: usize,
    confidence_threshold: f32,
    nms_threshold: f32,
}

impl Element for Preprocess {
    fn run(self: Box<Self>, io: ElementIo) -> Result<()> {
        loop {
            let packet = match io.recv("in")? {
                Some(packet) => packet,
                None => continue,
            };
            if packet.is_eos() {
                io.broadcast_eos()?;
                return Ok(());
            }
            let (tensor, letterbox) = preprocess_tensor(
                packet.tensor_ref().ok_or_else(|| {
                    Error::Runtime("yolo preprocess expects a tensor".to_string())
                })?,
                self.target_width,
                self.target_height,
            )?;
            let mut meta = packet.meta.clone();
            meta.tags
                .insert("yolo_scale".to_string(), letterbox.scale.to_string());
            meta.tags
                .insert("yolo_pad_x".to_string(), letterbox.pad_x.to_string());
            meta.tags
                .insert("yolo_pad_y".to_string(), letterbox.pad_y.to_string());
            meta.tags.insert(
                "yolo_source_width".to_string(),
                letterbox.source_width.to_string(),
            );
            meta.tags.insert(
                "yolo_source_height".to_string(),
                letterbox.source_height.to_string(),
            );
            io.send("out", Packet::tensor(tensor).with_meta(meta))?;
        }
    }
}

impl Element for Postprocess {
    fn run(self: Box<Self>, io: ElementIo) -> Result<()> {
        loop {
            let packet = match io.recv("in")? {
                Some(packet) => packet,
                None => continue,
            };
            if packet.is_eos() {
                io.broadcast_eos()?;
                return Ok(());
            }
            let tensor = packet
                .tensor_ref()
                .ok_or_else(|| Error::Runtime("yolo postprocess expects a tensor".to_string()))?;
            let detections = decode_detections(
                tensor,
                &packet.meta,
                self.target_width,
                self.target_height,
                self.class_count,
                self.confidence_threshold,
                self.nms_threshold,
            )?;
            io.send(
                "out",
                Packet::detections(detections).with_meta(packet.meta.clone()),
            )?;
        }
    }
}

fn create_preprocess(node: &NodeSpec) -> Result<CreatedElement> {
    let (target_width, target_height) = parse_preprocess(node)?;
    Ok(CreatedElement {
        element: Box::new(Preprocess {
            target_width,
            target_height,
        }),
        handle: ElementHandle::None,
    })
}

fn create_postprocess(node: &NodeSpec) -> Result<CreatedElement> {
    let config = parse_postprocess(node)?;
    Ok(CreatedElement {
        element: Box::new(config),
        handle: ElementHandle::None,
    })
}

fn validate_preprocess(node: &NodeSpec) -> Result<()> {
    parse_preprocess(node).map(|_| ())
}

fn validate_postprocess(node: &NodeSpec) -> Result<()> {
    parse_postprocess(node).map(|_| ())
}

fn parse_preprocess(node: &NodeSpec) -> Result<(usize, usize)> {
    let params = params_object(node)?;
    reject_unknown_fields(params, PREPROCESS_FIELDS)?;
    let target_width = read_nonzero_usize(params, "input_width", 640)?;
    let target_height = read_nonzero_usize(params, "input_height", target_width)?;
    target_width
        .checked_mul(target_height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| Error::Config("yolo input dimensions overflow".to_string()))?;
    Ok((target_width, target_height))
}

fn parse_postprocess(node: &NodeSpec) -> Result<Postprocess> {
    let params = params_object(node)?;
    reject_unknown_fields(params, POSTPROCESS_FIELDS)?;
    let input_width = read_usize(params, "input_width", 640)?;
    let input_height = read_usize(params, "input_height", input_width)?;
    ensure_nonzero(input_width, "input_width")?;
    ensure_nonzero(input_height, "input_height")?;
    let class_count = read_nonzero_usize(params, "class_count", 1)?;
    class_count
        .checked_add(5)
        .ok_or_else(|| Error::Config("class_count is too large".to_string()))?;
    let confidence_threshold = read_probability(params, "confidence_threshold", 0.25)?;
    let nms_threshold = read_probability(params, "nms_threshold", 0.45)?;
    Ok(Postprocess {
        target_width: usize_to_f32(input_width, "input_width")?,
        target_height: usize_to_f32(input_height, "input_height")?,
        class_count,
        confidence_threshold,
        nms_threshold,
    })
}

fn preprocess_tensor(
    input: &Tensor,
    target_width: usize,
    target_height: usize,
) -> Result<(Tensor, Letterbox)> {
    let dims = input.desc().shape().dims();
    let (channels, source_height, source_width, channel_first) = match (input.desc().format(), dims)
    {
        (DataFormat::NCHW, [1, channels, height, width]) => (*channels, *height, *width, true),
        (DataFormat::NCHW, [channels, height, width]) => (*channels, *height, *width, true),
        (DataFormat::NHWC, [1, height, width, channels]) => (*channels, *height, *width, false),
        (DataFormat::NHWC, [height, width, channels]) => (*channels, *height, *width, false),
        _ => {
            return Err(Error::Config(
                "yolo preprocess expects NCHW or NHWC rank 3/4 input".to_string(),
            ))
        }
    };
    if channels != 3 {
        return Err(Error::Config(
            "yolo preprocess currently expects three channels".to_string(),
        ));
    }
    let values = tensor_values(input)?;
    let expected = channels
        .checked_mul(source_height)
        .and_then(|size| size.checked_mul(source_width))
        .ok_or_else(|| Error::Runtime("input dimensions overflow".to_string()))?;
    if values.len() != expected {
        return Err(Error::Runtime(
            "input tensor size does not match its descriptor".to_string(),
        ));
    }
    let mut hwc = vec![0.0; expected];
    for y in 0..source_height {
        for x in 0..source_width {
            for channel in 0..channels {
                let source_index = if channel_first {
                    (channel * source_height + y) * source_width + x
                } else {
                    (y * source_width + x) * channels + channel
                };
                let target_index = (y * source_width + x) * channels + channel;
                hwc[target_index] = values[source_index];
            }
        }
    }
    let (resized, letterbox) = resize_letterbox(
        &hwc,
        channels,
        source_width,
        source_height,
        target_width,
        target_height,
        0.0,
    )
    .map_err(Error::Config)?;
    let device = dg_core::CpuDevice::new();
    let output_desc = TensorDesc::new(
        Shape::new([1, channels, target_height, target_width]),
        DataType::F32,
        DataFormat::NCHW,
        DeviceKind::Cpu,
    );
    let output = Tensor::allocate(&device, output_desc)?;
    let count = target_width
        .checked_mul(target_height)
        .and_then(|size| size.checked_mul(channels))
        .ok_or_else(|| Error::Runtime("output dimensions overflow".to_string()))?;
    let mut bytes = Vec::with_capacity(count * std::mem::size_of::<f32>());
    for channel in 0..channels {
        for y in 0..target_height {
            for x in 0..target_width {
                let value = resized[(y * target_width + x) * channels + channel] / 255.0;
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
        }
    }
    output.buffer().write_from_slice(&bytes)?;
    Ok((output, letterbox))
}

fn decode_detections(
    tensor: &Tensor,
    meta: &dg_graph::PacketMeta,
    target_width: f32,
    target_height: f32,
    class_count: usize,
    confidence_threshold: f32,
    nms_threshold: f32,
) -> Result<Vec<dg_core::Detection>> {
    let values = tensor_values(tensor)?;
    let attributes = class_count
        .checked_add(5)
        .ok_or_else(|| Error::Runtime("yolo attribute count overflow".to_string()))?;
    if attributes == 0 || values.len() % attributes != 0 {
        return Err(Error::Runtime(
            "yolo output size is not divisible by its attribute count".to_string(),
        ));
    }
    let scale = read_tag(meta, "yolo_scale")?;
    let pad_x = read_tag(meta, "yolo_pad_x")?;
    let pad_y = read_tag(meta, "yolo_pad_y")?;
    let source_width = read_tag(meta, "yolo_source_width")?;
    let source_height = read_tag(meta, "yolo_source_height")?;
    let letterbox = Letterbox {
        source_width: f32_to_usize(source_width)?,
        source_height: f32_to_usize(source_height)?,
        target_width: f32_to_usize(target_width)?,
        target_height: f32_to_usize(target_height)?,
        scale,
        pad_x,
        pad_y,
    };
    let mut detections = Vec::new();
    for row in values.chunks_exact(attributes) {
        let objectness = sigmoid(row[4]);
        let (class_id, class_logit) = row[5..]
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .ok_or_else(|| Error::Runtime("yolo class scores are empty".to_string()))?;
        let score = objectness * sigmoid(class_logit);
        if score < confidence_threshold {
            continue;
        }
        let width = row[2].exp().clamp(0.0, 2.0) * target_width;
        let height = row[3].exp().clamp(0.0, 2.0) * target_height;
        let bbox = dg_core::BBox::new(
            sigmoid(row[0]) * target_width - width * 0.5,
            sigmoid(row[1]) * target_height - height * 0.5,
            width,
            height,
        );
        let class_id = u32::try_from(class_id)
            .map_err(|_| Error::Runtime("yolo class id is out of range".to_string()))?;
        detections.push(dg_core::Detection::new(
            letterbox.map_to_source(bbox),
            score,
            class_id,
        ));
    }
    Ok(nms(&detections, nms_threshold))
}

fn tensor_values(tensor: &Tensor) -> Result<Vec<f32>> {
    let bytes = tensor.buffer().read_bytes();
    match tensor.desc().dtype() {
        DataType::U8 => Ok(bytes.into_iter().map(f32::from).collect()),
        DataType::F32 => {
            let chunks = bytes.chunks_exact(std::mem::size_of::<f32>());
            if !chunks.remainder().is_empty() {
                return Err(Error::Runtime(
                    "f32 tensor contains a partial element".to_string(),
                ));
            }
            chunks
                .map(|chunk| {
                    let array: [u8; 4] = chunk
                        .try_into()
                        .map_err(|_| Error::Runtime("invalid f32 tensor element".to_string()))?;
                    Ok(f32::from_ne_bytes(array))
                })
                .collect()
        }
        dtype => Err(Error::Config(format!(
            "yolo elements support only u8/f32 tensors, got {dtype:?}"
        ))),
    }
}

fn params_object(node: &NodeSpec) -> Result<&serde_json::Map<String, serde_json::Value>> {
    node.params
        .as_object()
        .ok_or_else(|| Error::Config(format!("node {} params must be an object", node.name)))
}

fn reject_unknown_fields(
    params: &serde_json::Map<String, serde_json::Value>,
    allowed: &[&str],
) -> Result<()> {
    for key in params.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(Error::Config(format!(
                "unknown field `{key}`; expected one of {}",
                allowed.join(", ")
            )));
        }
    }
    Ok(())
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
        usize::try_from(value).map_err(|_| Error::Config(format!("field {key} is out of range")))
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
            .map_err(|_| Error::Config(format!("field {key} cannot be represented as f32")))?;
        if !narrowed.is_finite() {
            return Err(Error::Config(format!(
                "field {key} cannot be represented as f32"
            )));
        }
        Ok(narrowed)
    })
}

fn read_nonzero_usize(
    params: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: usize,
) -> Result<usize> {
    let value = read_usize(params, key, default)?;
    ensure_nonzero(value, key)
}

fn ensure_nonzero(value: usize, key: &str) -> Result<usize> {
    if value == 0 {
        return Err(Error::Config(format!("field {key} must be non-zero")));
    }
    Ok(value)
}

fn read_probability(
    params: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: f32,
) -> Result<f32> {
    let value = read_f32(params, key, default)?;
    if !(0.0..=1.0).contains(&value) {
        return Err(Error::Config(format!(
            "field {key} must be between 0 and 1"
        )));
    }
    Ok(value)
}

fn read_tag(meta: &dg_graph::PacketMeta, key: &str) -> Result<f32> {
    meta.tags
        .get(key)
        .ok_or_else(|| Error::Runtime(format!("missing yolo metadata {key}")))?
        .parse::<f32>()
        .map_err(|_| Error::Runtime(format!("invalid yolo metadata {key}")))
}

fn usize_to_f32(value: usize, field: &str) -> Result<f32> {
    if value > 16_777_216 {
        return Err(Error::Config(format!(
            "{field} cannot be represented exactly as f32"
        )));
    }
    let value =
        u32::try_from(value).map_err(|_| Error::Config(format!("{field} is out of range")))?;
    value
        .to_string()
        .parse::<f32>()
        .map_err(|_| Error::Config(format!("{field} cannot be represented as f32")))
}

fn f32_to_usize(value: f32) -> Result<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return Err(Error::Runtime(
            "invalid yolo dimension metadata".to_string(),
        ));
    }
    value
        .to_string()
        .parse::<usize>()
        .map_err(|_| Error::Runtime("yolo dimension metadata is out of range".to_string()))
}
