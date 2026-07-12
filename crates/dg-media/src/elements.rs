//! Registered graph elements wrapping the Sans-I/O media cores.
//!
//! Each element is a thin driver: it moves packets between graph ports and a
//! core's submit/poll state machine. All media logic lives in [`crate::ops`].

use dg_graph::{
    CreatedElement, Element, ElementHandle, ElementIo, Error, NodeSpec, ParamField, ParamType,
    PortSchema, Result,
};
use serde_json::{Map, Value};
use tracing::trace;

#[cfg(feature = "avcodec")]
use crate::avcodec::{
    DecodeCore as AvcodecDecodeCore, EncodeCore as AvcodecEncodeCore,
    ResizeCore as AvcodecResizeCore,
};
use crate::bridge::{graph_packet_to_media_frame, media_frame_to_graph_packet};
use crate::ops::{DecodeCore, EncodeCore, MediaPoll, OsdBox, OsdCore, ResizeCore};
use crate::MediaFrame;

const MEDIA_INPUT: [PortSchema; 1] = [PortSchema {
    name: "in",
    dtype: None,
    required: true,
}];
const MEDIA_OUTPUT: [PortSchema; 1] = [PortSchema {
    name: "out",
    dtype: None,
    required: false,
}];
#[cfg(feature = "avcodec")]
const DECODE_PARAM_FIELDS: &[&str] = &["width", "height", "channels", "codec"];
#[cfg(not(feature = "avcodec"))]
const DECODE_PARAM_FIELDS: &[&str] = &["width", "height", "channels"];
#[cfg(feature = "avcodec")]
const ENCODE_PARAM_FIELDS: &[&str] = &["codec"];
const RESIZE_PARAM_FIELDS: &[&str] = &["width", "height"];
const OSD_PARAM_FIELDS: &[&str] = &["boxes", "color", "thickness"];
const OSD_BOX_FIELDS: &[&str] = &["x", "y", "width", "height"];
#[cfg(not(feature = "avcodec"))]
const EMPTY_PARAMS: &[ParamField] = &[];
const DECODE_PARAMS: &[ParamField] = &[
    ParamField {
        name: "width",
        ty: ParamType::Uint,
        required: true,
    },
    ParamField {
        name: "height",
        ty: ParamType::Uint,
        required: true,
    },
    ParamField {
        name: "channels",
        ty: ParamType::Uint,
        required: false,
    },
    #[cfg(feature = "avcodec")]
    ParamField {
        name: "codec",
        ty: ParamType::Enum(&["jpeg", "mjpeg", "h264"]),
        required: false,
    },
];
#[cfg(feature = "avcodec")]
const ENCODE_PARAMS: &[ParamField] = &[ParamField {
    name: "codec",
    ty: ParamType::Enum(&["jpeg", "mjpeg", "h264"]),
    required: false,
}];
#[cfg(not(feature = "avcodec"))]
const ENCODE_PARAMS: &[ParamField] = EMPTY_PARAMS;
const RESIZE_PARAMS: &[ParamField] = &[
    ParamField {
        name: "width",
        ty: ParamType::Uint,
        required: true,
    },
    ParamField {
        name: "height",
        ty: ParamType::Uint,
        required: true,
    },
];
const OSD_PARAMS: &[ParamField] = &[
    ParamField {
        name: "boxes",
        ty: ParamType::Array(&ParamType::Object),
        required: false,
    },
    ParamField {
        name: "color",
        ty: ParamType::Array(&ParamType::Uint),
        required: false,
    },
    ParamField {
        name: "thickness",
        ty: ParamType::Uint,
        required: false,
    },
];

inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "media_decode",
        input_ports: &MEDIA_INPUT,
        output_ports: &MEDIA_OUTPUT,
        params: DECODE_PARAMS,
        validate: Some(validate_decode),
        create: create_decode,
    }
}
inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "media_encode",
        input_ports: &MEDIA_INPUT,
        output_ports: &MEDIA_OUTPUT,
        params: ENCODE_PARAMS,
        validate: Some(validate_encode),
        create: create_encode,
    }
}
inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "media_resize",
        input_ports: &MEDIA_INPUT,
        output_ports: &MEDIA_OUTPUT,
        params: RESIZE_PARAMS,
        validate: Some(validate_resize),
        create: create_resize,
    }
}
inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "media_osd",
        input_ports: &MEDIA_INPUT,
        output_ports: &MEDIA_OUTPUT,
        params: OSD_PARAMS,
        validate: Some(validate_osd),
        create: create_osd,
    }
}

trait MediaCore: Send {
    fn submit(&mut self, frame: MediaFrame) -> dg_core::Result<()>;
    fn submit_end_of_stream(&mut self);
    fn poll(&mut self) -> core::result::Result<MediaPoll, dg_core::Error>;
}

impl MediaCore for DecodeCore {
    fn submit(&mut self, frame: MediaFrame) -> dg_core::Result<()> {
        self.submit_packet(frame)
    }
    fn submit_end_of_stream(&mut self) {
        Self::submit_end_of_stream(self);
    }
    fn poll(&mut self) -> core::result::Result<MediaPoll, dg_core::Error> {
        Ok(Self::poll(self))
    }
}

impl MediaCore for EncodeCore {
    fn submit(&mut self, frame: MediaFrame) -> dg_core::Result<()> {
        self.submit_image(frame)
    }
    fn submit_end_of_stream(&mut self) {
        Self::submit_end_of_stream(self);
    }
    fn poll(&mut self) -> core::result::Result<MediaPoll, dg_core::Error> {
        Ok(Self::poll(self))
    }
}

impl MediaCore for ResizeCore {
    fn submit(&mut self, frame: MediaFrame) -> dg_core::Result<()> {
        self.submit_image(frame)
    }
    fn submit_end_of_stream(&mut self) {
        Self::submit_end_of_stream(self);
    }
    fn poll(&mut self) -> core::result::Result<MediaPoll, dg_core::Error> {
        Ok(Self::poll(self))
    }
}

impl MediaCore for OsdCore {
    fn submit(&mut self, frame: MediaFrame) -> dg_core::Result<()> {
        self.submit_image(frame)
    }
    fn submit_end_of_stream(&mut self) {
        Self::submit_end_of_stream(self);
    }
    fn poll(&mut self) -> core::result::Result<MediaPoll, dg_core::Error> {
        Ok(Self::poll(self))
    }
}

#[cfg(feature = "avcodec")]
impl MediaCore for AvcodecDecodeCore {
    fn submit(&mut self, frame: MediaFrame) -> dg_core::Result<()> {
        self.submit_packet(frame)
    }

    fn submit_end_of_stream(&mut self) {
        Self::submit_end_of_stream(self);
    }

    fn poll(&mut self) -> core::result::Result<MediaPoll, dg_core::Error> {
        Self::poll(self)
    }
}

#[cfg(feature = "avcodec")]
impl MediaCore for AvcodecEncodeCore {
    fn submit(&mut self, frame: MediaFrame) -> dg_core::Result<()> {
        self.submit_image(frame)
    }

    fn submit_end_of_stream(&mut self) {
        Self::submit_end_of_stream(self);
    }

    fn poll(&mut self) -> core::result::Result<MediaPoll, dg_core::Error> {
        Self::poll(self)
    }
}

#[cfg(feature = "avcodec")]
impl MediaCore for AvcodecResizeCore {
    fn submit(&mut self, frame: MediaFrame) -> dg_core::Result<()> {
        self.submit_image(frame)
    }

    fn submit_end_of_stream(&mut self) {
        Self::submit_end_of_stream(self);
    }

    fn poll(&mut self) -> core::result::Result<MediaPoll, dg_core::Error> {
        Self::poll(self)
    }
}

struct MediaElement<C: MediaCore> {
    core: C,
}

impl<C: MediaCore> MediaElement<C> {
    fn drain(&mut self, io: &ElementIo) -> Result<bool> {
        loop {
            match self.core.poll().map_err(|err| Error::Element {
                element: io.name.clone(),
                message: err.to_string(),
            })? {
                MediaPoll::Ready(frame) => {
                    let meta = dg_graph::PacketMeta {
                        sequence: frame
                            .meta
                            .pts
                            .and_then(|pts| pts.try_into().ok())
                            .unwrap_or(0),
                        stream_id: frame.meta.stream_id.clone(),
                        tags: frame.meta.tags.clone(),
                    };
                    let packet = media_frame_to_graph_packet(frame)?.with_meta(meta);
                    io.send("out", packet)?;
                }
                MediaPoll::Pending => return Ok(false),
                MediaPoll::EndOfStream => return Ok(true),
            }
        }
    }
}

impl<C: MediaCore> Element for MediaElement<C> {
    fn run(mut self: Box<Self>, io: ElementIo) -> Result<()> {
        trace!(node = %io.name, "running media element");
        loop {
            let packet = match io.recv("in") {
                Ok(Some(packet)) => packet,
                Ok(None) => {
                    if io.stop.load(std::sync::atomic::Ordering::Relaxed) {
                        return Err(Error::NotRunning);
                    }
                    continue;
                }
                Err(err) => return Err(err),
            };
            if packet.is_eos() {
                self.core.submit_end_of_stream();
                if self.drain(&io)? {
                    io.broadcast_eos()?;
                    return Ok(());
                }
                return Err(Error::Runtime(
                    "media element did not reach end of stream after eos".to_string(),
                ));
            }
            let frame = graph_packet_to_media_frame(packet);
            self.core.submit(frame).map_err(|err| Error::Element {
                element: io.name.clone(),
                message: err.to_string(),
            })?;
            self.drain(&io)?;
        }
    }
}

fn create_decode(node: &NodeSpec) -> Result<CreatedElement> {
    let (width, height, channels) = parse_decode(node)?;
    #[cfg(feature = "avcodec")]
    let codec = parse_codec(node)?;
    #[cfg(feature = "avcodec")]
    let core = {
        let _ = (width, height, channels);
        AvcodecDecodeCore::new(codec)?
    };
    #[cfg(not(feature = "avcodec"))]
    let core = DecodeCore::new(width, height, channels);
    Ok(CreatedElement {
        element: Box::new(MediaElement { core }),
        handle: ElementHandle::None,
    })
}

fn create_encode(node: &NodeSpec) -> Result<CreatedElement> {
    validate_encode(node)?;
    #[cfg(feature = "avcodec")]
    let codec = if node.params.is_null() {
        crate::avcodec::codec_from_name(None).map_err(|err| Error::Config(err.to_string()))?
    } else {
        parse_codec(node)?
    };
    #[cfg(feature = "avcodec")]
    let core = AvcodecEncodeCore::new(codec)?;
    #[cfg(not(feature = "avcodec"))]
    let core = EncodeCore::new();
    Ok(CreatedElement {
        element: Box::new(MediaElement { core }),
        handle: ElementHandle::None,
    })
}

fn create_resize(node: &NodeSpec) -> Result<CreatedElement> {
    let (width, height) = parse_resize(node)?;
    #[cfg(feature = "avcodec")]
    let core = AvcodecResizeCore::new(width, height)?;
    #[cfg(not(feature = "avcodec"))]
    let core = ResizeCore::new(width, height);
    Ok(CreatedElement {
        element: Box::new(MediaElement { core }),
        handle: ElementHandle::None,
    })
}

fn create_osd(node: &NodeSpec) -> Result<CreatedElement> {
    let (boxes, color, thickness) = parse_osd(node)?;
    Ok(CreatedElement {
        element: Box::new(MediaElement {
            core: OsdCore::new(boxes, color, thickness),
        }),
        handle: ElementHandle::None,
    })
}

fn validate_decode(node: &NodeSpec) -> Result<()> {
    parse_decode(node)?;
    #[cfg(feature = "avcodec")]
    parse_codec(node)?;
    Ok(())
}

fn validate_encode(node: &NodeSpec) -> Result<()> {
    if node.params.is_null() {
        return Ok(());
    }
    let params = params_object(node)?;
    #[cfg(feature = "avcodec")]
    {
        reject_unknown_fields(params, ENCODE_PARAM_FIELDS)?;
        parse_codec(node)?;
    }
    #[cfg(not(feature = "avcodec"))]
    reject_unknown_fields(params, &[])?;
    Ok(())
}

fn parse_decode(node: &NodeSpec) -> Result<(usize, usize, usize)> {
    let params = params_object(node)?;
    reject_unknown_fields(params, DECODE_PARAM_FIELDS)?;
    let width = required_nonzero(params, "width", &node.name)?;
    let height = required_nonzero(params, "height", &node.name)?;
    let channels = read_usize(params, "channels")?.unwrap_or(3);
    ensure_nonzero(channels, "channels")?;
    height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| Error::Config("image dimensions overflow".to_string()))?;
    Ok((width, height, channels))
}

#[cfg(feature = "avcodec")]
fn parse_codec(node: &NodeSpec) -> Result<dg_media_avcodec::CodecId> {
    let params = params_object(node)?;
    let name = params.get("codec").and_then(Value::as_str);
    crate::avcodec::codec_from_name(name).map_err(|err| Error::Config(err.to_string()))
}

fn validate_resize(node: &NodeSpec) -> Result<()> {
    parse_resize(node).map(|_| ())
}

fn parse_resize(node: &NodeSpec) -> Result<(usize, usize)> {
    let params = params_object(node)?;
    reject_unknown_fields(params, RESIZE_PARAM_FIELDS)?;
    let width = required_nonzero(params, "width", &node.name)?;
    let height = required_nonzero(params, "height", &node.name)?;
    height
        .checked_mul(width)
        .ok_or_else(|| Error::Config("image dimensions overflow".to_string()))?;
    Ok((width, height))
}

fn validate_osd(node: &NodeSpec) -> Result<()> {
    parse_osd(node).map(|_| ())
}

fn parse_osd(node: &NodeSpec) -> Result<(Vec<OsdBox>, Vec<u8>, usize)> {
    let params = params_object(node)?;
    reject_unknown_fields(params, OSD_PARAM_FIELDS)?;
    let boxes = read_boxes(params, &node.name)?;
    let color = read_u8_array(params, "color")?.unwrap_or_else(|| vec![255, 0, 0]);
    if color.is_empty() {
        return Err(Error::Config("field color must not be empty".to_string()));
    }
    let thickness = read_usize(params, "thickness")?.unwrap_or(1);
    ensure_nonzero(thickness, "thickness")?;
    Ok((boxes, color, thickness))
}

fn params_object(node: &NodeSpec) -> Result<&Map<String, Value>> {
    node.params
        .as_object()
        .ok_or_else(|| Error::Config(format!("node {} params must be an object", node.name)))
}

fn reject_unknown_fields(params: &Map<String, Value>, allowed: &[&str]) -> Result<()> {
    for key in params.keys() {
        if !allowed.contains(&key.as_str()) {
            let message = if allowed.is_empty() {
                format!("unknown field `{key}`; no parameters are supported")
            } else {
                format!(
                    "unknown field `{key}`; expected one of {}",
                    allowed.join(", ")
                )
            };
            return Err(Error::Config(message));
        }
    }
    Ok(())
}

fn required_nonzero(params: &Map<String, Value>, key: &str, node: &str) -> Result<usize> {
    let value = read_usize(params, key)?
        .ok_or_else(|| Error::Config(format!("node {node}: field {key} is required")))?;
    ensure_nonzero(value, key)
}

fn ensure_nonzero(value: usize, key: &str) -> Result<usize> {
    if value == 0 {
        return Err(Error::Config(format!("field {key} must be non-zero")));
    }
    Ok(value)
}

fn read_usize(params: &Map<String, Value>, key: &str) -> Result<Option<usize>> {
    match params.get(key) {
        Some(value) => value
            .as_u64()
            .ok_or_else(|| Error::Config(format!("field {key} must be a non-negative integer")))
            .and_then(|value| {
                usize::try_from(value).map_err(|_| Error::Config(format!("field {key} overflow")))
            })
            .map(Some),
        None => Ok(None),
    }
}

fn read_u8_array(params: &Map<String, Value>, key: &str) -> Result<Option<Vec<u8>>> {
    match params.get(key) {
        Some(value) => {
            let array = value
                .as_array()
                .ok_or_else(|| Error::Config(format!("field {key} must be an array")))?;
            let values = array
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .ok_or_else(|| Error::Config(format!("field {key} must contain integers")))
                        .and_then(|v| {
                            u8::try_from(v)
                                .map_err(|_| Error::Config(format!("field {key} overflow")))
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Some(values))
        }
        None => Ok(None),
    }
}

fn read_boxes(params: &Map<String, Value>, node: &str) -> Result<Vec<OsdBox>> {
    let Some(value) = params.get("boxes") else {
        return Ok(Vec::new());
    };
    let array = value
        .as_array()
        .ok_or_else(|| Error::Config(format!("node {node}: field boxes must be an array")))?;
    array
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let object = entry
                .as_object()
                .ok_or_else(|| Error::Config(format!("node {node}: each box must be an object")))?;
            reject_unknown_fields(object, OSD_BOX_FIELDS).map_err(|err| match err {
                Error::Config(message) => {
                    Error::Config(format!("node {node}: field boxes[{index}]: {message}"))
                }
                other => other,
            })?;
            let field = |key: &str| -> Result<usize> {
                read_usize(object, key)?.ok_or_else(|| {
                    Error::Config(format!("node {node}: box field {key} is required"))
                })
            };
            let x = field("x")?;
            let y = field("y")?;
            let width = ensure_nonzero(field("width")?, "boxes[].width")?;
            let height = ensure_nonzero(field("height")?, "boxes[].height")?;
            x.checked_add(width)
                .ok_or_else(|| Error::Config("box horizontal extent overflow".to_string()))?;
            y.checked_add(height)
                .ok_or_else(|| Error::Config("box vertical extent overflow".to_string()))?;
            Ok(OsdBox {
                x,
                y,
                width,
                height,
            })
        })
        .collect()
}
