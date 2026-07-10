//! Graph source/sink elements for stream pull (RTSP / HTTP-FLV) and push
//! (RTMP / WebRTC) endpoints.
//!
//! Elements are registered into the `dg-graph` element inventory under the
//! kinds `rtsp_src`, `httpflv_src`, `rtmp_sink`, and `webrtc_sink`. URL scheme
//! selection is delegated to [`crate::connector`]: `mock://` runs fully
//! in-process, protocol schemes require the feature-gated cheetah runtime.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use dg_core::DataType;
use dg_graph::{
    CreatedElement, Element, ElementDescriptor, ElementHandle, ElementIo, NodeSpec, Packet,
    PacketMeta, PortSchema,
};
use serde_json::{Map, Value};
use tracing::debug;

use crate::connector::{open_pull, open_push, PullEndpoint, StreamProtocol};
use crate::hub::MEDIA_TAG;
use crate::stream::SubscriberSourceSyncExt;
use crate::stream::{
    BackpressurePolicy, DispatchResult, MediaFilter, PublisherOptions, PublisherSink,
    SubscriberOptions,
};
use crate::track::{TrackInfo, TrackReadiness};
use dg_media::{MediaFrame, MediaFrameKind};

const PULL_OUTPUT_PORT: PortSchema = PortSchema {
    name: "out",
    dtype: Some(DataType::U8),
};
const PUSH_INPUT_PORT: PortSchema = PortSchema {
    name: "in",
    dtype: None,
};

const PTS_TAG: &str = "pts";
const DTS_TAG: &str = "dts";

inventory::submit! {
    ElementDescriptor {
        kind: "rtsp_src",
        input_ports: &[],
        output_ports: &[PULL_OUTPUT_PORT],
        create: create_rtsp_src,
    }
}

inventory::submit! {
    ElementDescriptor {
        kind: "httpflv_src",
        input_ports: &[],
        output_ports: &[PULL_OUTPUT_PORT],
        create: create_httpflv_src,
    }
}

inventory::submit! {
    ElementDescriptor {
        kind: "rtmp_sink",
        input_ports: &[PUSH_INPUT_PORT],
        output_ports: &[],
        create: create_rtmp_sink,
    }
}

inventory::submit! {
    ElementDescriptor {
        kind: "webrtc_sink",
        input_ports: &[PUSH_INPUT_PORT],
        output_ports: &[],
        create: create_webrtc_sink,
    }
}

struct StreamPullElement {
    endpoint: PullEndpoint,
}

impl Element for StreamPullElement {
    fn run(mut self: Box<Self>, io: ElementIo) -> dg_graph::Result<()> {
        for track in &self.endpoint.tracks {
            if track.readiness != TrackReadiness::Ready {
                let _ = self.endpoint.source.close_blocking();
                return Err(dg_graph::Error::Runtime(format!(
                    "track {} is not ready ({:?})",
                    track.track_id, track.readiness
                )));
            }
            if let Err(err) = track.validate_codec_config() {
                let _ = self.endpoint.source.close_blocking();
                return Err(dg_graph::Error::Runtime(format!(
                    "track codec config invalid: {err}"
                )));
            }
        }
        let mut sequence = 0u64;
        loop {
            if io.stop.load(Ordering::Relaxed) {
                let _ = self.endpoint.source.close_blocking();
                return Err(dg_graph::Error::NotRunning);
            }
            match self.endpoint.source.recv_blocking() {
                Ok(Some(frame)) if frame.is_end_of_stream() => break,
                Ok(Some(frame)) => {
                    let packet = media_frame_to_packet(&frame, sequence)?;
                    sequence = sequence.saturating_add(1);
                    io.send("out", packet)?;
                }
                Ok(None) => break,
                Err(err) => {
                    let _ = self.endpoint.source.close_blocking();
                    return Err(dg_graph::Error::Runtime(format!(
                        "stream source error: {err}"
                    )));
                }
            }
        }
        let _ = self.endpoint.source.close_blocking();
        io.broadcast_eos()
    }
}

struct StreamPushElement {
    sink: Box<dyn PublisherSink>,
    tracks: Vec<TrackInfo>,
    announce_tracks: bool,
}

impl Element for StreamPushElement {
    fn run(self: Box<Self>, io: ElementIo) -> dg_graph::Result<()> {
        if self.announce_tracks && !self.tracks.is_empty() {
            for track in &self.tracks {
                if track.readiness == TrackReadiness::Ready {
                    track.validate_codec_config().map_err(|err| {
                        dg_graph::Error::Runtime(format!("track codec config invalid: {err}"))
                    })?;
                }
            }
            self.sink
                .update_tracks(self.tracks.clone())
                .map_err(|err| {
                    dg_graph::Error::Runtime(format!("track announcement failed: {err}"))
                })?;
        }
        loop {
            let packet = match io.recv("in") {
                Ok(Some(packet)) => packet,
                Ok(None) => {
                    if io.stop.load(Ordering::Relaxed) {
                        let _ = self.sink.close();
                        return Err(dg_graph::Error::NotRunning);
                    }
                    continue;
                }
                Err(err) => {
                    let _ = self.sink.close();
                    return Err(err);
                }
            };
            if packet.is_eos() {
                self.sink.close().map_err(|err| {
                    dg_graph::Error::Runtime(format!("publisher close failed: {err}"))
                })?;
                return Ok(());
            }
            let frame = packet_to_media_frame(packet)?;
            match self.sink.push_frame(Arc::new(frame)) {
                Ok(DispatchResult::Accepted) => {}
                Ok(DispatchResult::DroppedByPolicy) => {
                    debug!(node = %io.name, "frame dropped by backpressure policy");
                }
                Ok(DispatchResult::RejectedClosed) => {
                    return Err(dg_graph::Error::Runtime(
                        "publisher rejected frame: stream closed".to_string(),
                    ));
                }
                Err(err) => {
                    let _ = self.sink.close();
                    return Err(dg_graph::Error::Runtime(format!(
                        "stream sink error: {err}"
                    )));
                }
            }
            let keyframe_requests = self.sink.take_keyframe_requests();
            if keyframe_requests > 0 {
                debug!(node = %io.name, keyframe_requests, "keyframe requested by remote peer");
            }
        }
    }
}

fn media_frame_to_packet(frame: &Arc<MediaFrame>, sequence: u64) -> dg_graph::Result<Packet> {
    let mut frame = match Arc::try_unwrap(Arc::clone(frame)) {
        Ok(frame) => frame,
        Err(shared) => shared.as_ref().clone(),
    };
    if frame.shape.is_empty() {
        frame.shape = vec![frame.buffer.len()];
    }
    let mut tags = frame.meta.tags.clone();
    if let Some(pts) = frame.meta.pts {
        tags.insert(PTS_TAG.to_string(), pts.to_string());
    }
    if let Some(dts) = frame.meta.dts {
        tags.insert(DTS_TAG.to_string(), dts.to_string());
    }
    let meta = PacketMeta {
        sequence,
        stream_id: frame.meta.stream_id.clone(),
        tags,
    };
    let tensor = frame.into_tensor()?;
    Ok(Packet::tensor(tensor).with_meta(meta))
}

fn packet_to_media_frame(packet: Packet) -> dg_graph::Result<MediaFrame> {
    let meta = packet.meta.clone();
    let tensor = packet
        .into_tensor()
        .ok_or_else(|| dg_graph::Error::Runtime("expected tensor payload".to_string()))?;
    let mut frame = MediaFrame::from_tensor(tensor);
    if meta.tags.get(MEDIA_TAG).map(String::as_str) == Some("video") {
        frame.kind = MediaFrameKind::Image;
    }
    frame.meta.pts = meta
        .tags
        .get(PTS_TAG)
        .and_then(|value| value.parse::<i64>().ok());
    frame.meta.dts = meta
        .tags
        .get(DTS_TAG)
        .and_then(|value| value.parse::<i64>().ok());
    frame.meta.stream_id = meta.stream_id;
    frame.meta.tags = meta.tags;
    Ok(frame)
}

fn create_rtsp_src(node: &NodeSpec) -> dg_graph::Result<CreatedElement> {
    create_pull(node, StreamProtocol::RtspPull)
}

fn create_httpflv_src(node: &NodeSpec) -> dg_graph::Result<CreatedElement> {
    create_pull(node, StreamProtocol::HttpFlvPull)
}

fn create_rtmp_sink(node: &NodeSpec) -> dg_graph::Result<CreatedElement> {
    create_push(node, StreamProtocol::RtmpPush)
}

fn create_webrtc_sink(node: &NodeSpec) -> dg_graph::Result<CreatedElement> {
    create_push(node, StreamProtocol::WebRtcPush)
}

fn create_pull(node: &NodeSpec, protocol: StreamProtocol) -> dg_graph::Result<CreatedElement> {
    let params = params_object(node)?;
    let url = read_url(params, node)?;
    let options = SubscriberOptions {
        queue_capacity: read_usize(params, "queue_capacity", 150)?,
        backpressure: read_backpressure(params)?,
        media_filter: MediaFilter {
            enable_video: read_bool(params, "enable_video", true)?,
            enable_audio: read_bool(params, "enable_audio", true)?,
        },
        ..SubscriberOptions::default()
    };
    let endpoint = open_pull(protocol, &url, options).map_err(create_error)?;
    Ok(CreatedElement {
        element: Box::new(StreamPullElement { endpoint }),
        handle: ElementHandle::None,
    })
}

fn create_push(node: &NodeSpec, protocol: StreamProtocol) -> dg_graph::Result<CreatedElement> {
    let params = params_object(node)?;
    let url = read_url(params, node)?;
    let announce_tracks = read_bool(params, "announce_tracks", true)?;
    let tracks = read_tracks(params)?;
    let options = PublisherOptions { announce_tracks };
    let sink = open_push(protocol, &url, options).map_err(create_error)?;
    Ok(CreatedElement {
        element: Box::new(StreamPushElement {
            sink,
            tracks,
            announce_tracks,
        }),
        handle: ElementHandle::None,
    })
}

fn create_error(err: crate::error::Error) -> dg_graph::Error {
    match err {
        crate::error::Error::InvalidArgument(message) => dg_graph::Error::Config(message),
        other => dg_graph::Error::Runtime(other.to_string()),
    }
}

fn params_object(node: &NodeSpec) -> dg_graph::Result<&Map<String, Value>> {
    node.params.as_object().ok_or_else(|| {
        dg_graph::Error::Config(format!("node {} params must be an object", node.name))
    })
}

fn read_url(params: &Map<String, Value>, node: &NodeSpec) -> dg_graph::Result<String> {
    params
        .get("url")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            dg_graph::Error::Config(format!(
                "node {} requires a string `url` parameter",
                node.name
            ))
        })
}

fn read_usize(params: &Map<String, Value>, key: &str, default: usize) -> dg_graph::Result<usize> {
    match params.get(key) {
        Some(value) => value
            .as_u64()
            .ok_or_else(|| {
                dg_graph::Error::Config(format!("field {key} must be a non-negative integer"))
            })
            .and_then(|value| {
                usize::try_from(value)
                    .map_err(|_| dg_graph::Error::Config(format!("field {key} overflow")))
            }),
        None => Ok(default),
    }
}

fn read_bool(params: &Map<String, Value>, key: &str, default: bool) -> dg_graph::Result<bool> {
    match params.get(key) {
        Some(value) => value
            .as_bool()
            .ok_or_else(|| dg_graph::Error::Config(format!("field {key} must be a boolean"))),
        None => Ok(default),
    }
}

fn read_backpressure(params: &Map<String, Value>) -> dg_graph::Result<BackpressurePolicy> {
    match params.get("backpressure") {
        None => Ok(BackpressurePolicy::DropDroppableFirst),
        Some(value) => match value.as_str() {
            Some("drop_droppable_first") => Ok(BackpressurePolicy::DropDroppableFirst),
            Some("drop_until_next_keyframe") => Ok(BackpressurePolicy::DropUntilNextKeyframe),
            Some("disconnect_on_overflow") => Ok(BackpressurePolicy::DisconnectOnOverflow),
            _ => Err(dg_graph::Error::Config(
                "field backpressure must be one of drop_droppable_first, \
                 drop_until_next_keyframe, disconnect_on_overflow"
                    .to_string(),
            )),
        },
    }
}

fn read_tracks(params: &Map<String, Value>) -> dg_graph::Result<Vec<TrackInfo>> {
    match params.get("tracks") {
        None => Ok(Vec::new()),
        Some(value) => serde_json::from_value(value.clone())
            .map_err(|err| dg_graph::Error::Config(format!("field tracks is invalid: {err}"))),
    }
}
