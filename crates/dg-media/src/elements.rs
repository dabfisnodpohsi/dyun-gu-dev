//! Registered graph elements wrapping the Sans-I/O media cores.
//!
//! Each element is a thin driver: it moves packets between graph ports and a
//! core's submit/poll state machine. All media logic lives in [`crate::ops`].

use dg_graph::{
    CreatedElement, Element, ElementHandle, ElementIo, Error, NodeSpec, PortSchema, Result,
};
use serde_json::{Map, Value};
use tracing::trace;

use crate::bridge::{graph_packet_to_media_frame, media_frame_to_graph_packet};
use crate::ops::{DecodeCore, EncodeCore, MediaPoll, OsdBox, OsdCore, ResizeCore};
use crate::MediaFrame;

const MEDIA_INPUT: [PortSchema; 1] = [PortSchema {
    name: "in",
    dtype: None,
}];
const MEDIA_OUTPUT: [PortSchema; 1] = [PortSchema {
    name: "out",
    dtype: None,
}];

inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "media_decode",
        input_ports: &MEDIA_INPUT,
        output_ports: &MEDIA_OUTPUT,
        create: create_decode,
    }
}
inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "media_encode",
        input_ports: &MEDIA_INPUT,
        output_ports: &MEDIA_OUTPUT,
        create: create_encode,
    }
}
inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "media_resize",
        input_ports: &MEDIA_INPUT,
        output_ports: &MEDIA_OUTPUT,
        create: create_resize,
    }
}
inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "media_osd",
        input_ports: &MEDIA_INPUT,
        output_ports: &MEDIA_OUTPUT,
        create: create_osd,
    }
}

trait MediaCore: Send {
    fn submit(&mut self, frame: MediaFrame) -> dg_core::Result<()>;
    fn submit_end_of_stream(&mut self);
    fn poll(&mut self) -> MediaPoll;
}

impl MediaCore for DecodeCore {
    fn submit(&mut self, frame: MediaFrame) -> dg_core::Result<()> {
        self.submit_packet(frame)
    }
    fn submit_end_of_stream(&mut self) {
        Self::submit_end_of_stream(self);
    }
    fn poll(&mut self) -> MediaPoll {
        Self::poll(self)
    }
}

impl MediaCore for EncodeCore {
    fn submit(&mut self, frame: MediaFrame) -> dg_core::Result<()> {
        self.submit_image(frame)
    }
    fn submit_end_of_stream(&mut self) {
        Self::submit_end_of_stream(self);
    }
    fn poll(&mut self) -> MediaPoll {
        Self::poll(self)
    }
}

impl MediaCore for ResizeCore {
    fn submit(&mut self, frame: MediaFrame) -> dg_core::Result<()> {
        self.submit_image(frame)
    }
    fn submit_end_of_stream(&mut self) {
        Self::submit_end_of_stream(self);
    }
    fn poll(&mut self) -> MediaPoll {
        Self::poll(self)
    }
}

impl MediaCore for OsdCore {
    fn submit(&mut self, frame: MediaFrame) -> dg_core::Result<()> {
        self.submit_image(frame)
    }
    fn submit_end_of_stream(&mut self) {
        Self::submit_end_of_stream(self);
    }
    fn poll(&mut self) -> MediaPoll {
        Self::poll(self)
    }
}

struct MediaElement<C: MediaCore> {
    core: C,
}

impl<C: MediaCore> MediaElement<C> {
    fn drain(&mut self, io: &ElementIo) -> Result<bool> {
        loop {
            match self.core.poll() {
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
    let params = params_object(node)?;
    let width = read_usize(params, "width")?
        .ok_or_else(|| Error::Config(format!("node {}: field width is required", node.name)))?;
    let height = read_usize(params, "height")?
        .ok_or_else(|| Error::Config(format!("node {}: field height is required", node.name)))?;
    let channels = read_usize(params, "channels")?.unwrap_or(3);
    Ok(CreatedElement {
        element: Box::new(MediaElement {
            core: DecodeCore::new(width, height, channels),
        }),
        handle: ElementHandle::None,
    })
}

fn create_encode(node: &NodeSpec) -> Result<CreatedElement> {
    let _ = params_object(node)?;
    Ok(CreatedElement {
        element: Box::new(MediaElement {
            core: EncodeCore::new(),
        }),
        handle: ElementHandle::None,
    })
}

fn create_resize(node: &NodeSpec) -> Result<CreatedElement> {
    let params = params_object(node)?;
    let width = read_usize(params, "width")?
        .ok_or_else(|| Error::Config(format!("node {}: field width is required", node.name)))?;
    let height = read_usize(params, "height")?
        .ok_or_else(|| Error::Config(format!("node {}: field height is required", node.name)))?;
    Ok(CreatedElement {
        element: Box::new(MediaElement {
            core: ResizeCore::new(width, height),
        }),
        handle: ElementHandle::None,
    })
}

fn create_osd(node: &NodeSpec) -> Result<CreatedElement> {
    let params = params_object(node)?;
    let boxes = read_boxes(params, &node.name)?;
    let color = read_u8_array(params, "color")?.unwrap_or_else(|| vec![255, 0, 0]);
    let thickness = read_usize(params, "thickness")?.unwrap_or(1);
    Ok(CreatedElement {
        element: Box::new(MediaElement {
            core: OsdCore::new(boxes, color, thickness),
        }),
        handle: ElementHandle::None,
    })
}

fn params_object(node: &NodeSpec) -> Result<&Map<String, Value>> {
    node.params
        .as_object()
        .ok_or_else(|| Error::Config(format!("node {} params must be an object", node.name)))
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
        .map(|entry| {
            let object = entry
                .as_object()
                .ok_or_else(|| Error::Config(format!("node {node}: each box must be an object")))?;
            let field = |key: &str| -> Result<usize> {
                read_usize(object, key)?.ok_or_else(|| {
                    Error::Config(format!("node {node}: box field {key} is required"))
                })
            };
            Ok(OsdBox {
                x: field("x")?,
                y: field("y")?,
                width: field("width")?,
                height: field("height")?,
            })
        })
        .collect()
}
