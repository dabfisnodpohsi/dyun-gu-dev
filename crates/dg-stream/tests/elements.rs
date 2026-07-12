//! End-to-end tests for the registered stream elements and the mock hub.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use bytes::Bytes;
use dg_core::{DataFormat, DataType, DeviceKind};
use dg_graph::{find_element, Graph, GraphSpecBuilder, NodeSpec};
use dg_media::{MediaFrame, MediaFrameKind};
use dg_stream::{
    open_pull, open_push, BackpressurePolicy, CodecExtradata, CodecId, DispatchResult, Error,
    MediaKind, MemoryStreamHub, PublisherOptions, PublisherSink, StreamProtocol, SubscriberOptions,
    SubscriberSourceSyncExt, TrackInfo, TrackReadiness, KEYFRAME_TAG, MEDIA_TAG,
};

fn video_frame(pts: i64, keyframe: bool, payload: &[u8]) -> Arc<MediaFrame> {
    let mut frame = MediaFrame::from_host_bytes(
        MediaFrameKind::Image,
        DataType::U8,
        DataFormat::Auto,
        vec![payload.len()],
        DeviceKind::Cpu,
        payload.to_vec(),
    )
    .expect("frame");
    frame.meta.pts = Some(pts);
    frame.meta.dts = Some(pts);
    frame
        .meta
        .tags
        .insert(MEDIA_TAG.to_string(), "video".to_string());
    if keyframe {
        frame
            .meta
            .tags
            .insert(KEYFRAME_TAG.to_string(), "true".to_string());
    }
    Arc::new(frame)
}

fn h264_track(readiness: TrackReadiness, extradata: CodecExtradata) -> TrackInfo {
    let mut track = TrackInfo::new(1, MediaKind::Video, CodecId::H264, 90_000);
    track.extradata = extradata;
    track.readiness = readiness;
    track
}

fn h264_extradata() -> CodecExtradata {
    CodecExtradata::H264 {
        sps: vec![Bytes::from_static(&[0x67, 0x42])],
        pps: vec![Bytes::from_static(&[0x68, 0xce])],
        avcc: None,
    }
}

fn node(name: &str, kind: &str, params: serde_json::Value) -> NodeSpec {
    NodeSpec {
        name: name.to_string(),
        kind: kind.to_string(),
        template: None,
        params,
        ..NodeSpec::default()
    }
}

fn subscriber_options(capacity: usize, policy: BackpressurePolicy) -> SubscriberOptions {
    SubscriberOptions {
        queue_capacity: capacity,
        backpressure: policy,
        ..SubscriberOptions::default()
    }
}

#[test]
fn codec_config_validation() {
    let ready = h264_track(TrackReadiness::Ready, h264_extradata());
    assert!(ready.validate_codec_config().is_ok());
    assert!(ready.is_streamable());

    let missing = h264_track(TrackReadiness::Ready, CodecExtradata::None);
    assert!(missing.validate_codec_config().is_err());
    assert!(!missing.is_streamable());

    let mjpeg = TrackInfo::new(2, MediaKind::Video, CodecId::MJPEG, 90_000);
    assert!(mjpeg.validate_codec_config().is_ok());

    let pending = h264_track(TrackReadiness::PendingConfig, CodecExtradata::None);
    assert!(!pending.is_streamable());
}

#[test]
fn hub_propagates_tracks_and_close() {
    let hub = MemoryStreamHub::new();
    let publisher = hub
        .publish("mock://tracks", PublisherOptions::default())
        .expect("publish");
    publisher
        .update_tracks(vec![h264_track(TrackReadiness::Ready, h264_extradata())])
        .expect("tracks");
    assert_eq!(hub.tracks("mock://tracks").len(), 1);

    let mut subscriber = hub
        .subscribe(
            "mock://tracks",
            subscriber_options(8, BackpressurePolicy::DropDroppableFirst),
        )
        .expect("subscribe");
    assert_eq!(
        publisher
            .push_frame(video_frame(0, true, b"key"))
            .expect("push"),
        DispatchResult::Accepted
    );
    publisher.close().expect("close");
    assert_eq!(
        publisher
            .push_frame(video_frame(40, false, b"late"))
            .expect("push after close"),
        DispatchResult::RejectedClosed
    );

    let frame = subscriber.recv_blocking().expect("recv").expect("frame");
    assert_eq!(frame.meta.pts, Some(0));
    assert!(subscriber.recv_blocking().expect("recv").is_none());
}

#[test]
fn hub_rejects_ready_track_without_config() {
    let hub = MemoryStreamHub::new();
    let publisher = hub
        .publish("mock://invalid-track", PublisherOptions::default())
        .expect("publish");
    let err = publisher
        .update_tracks(vec![h264_track(
            TrackReadiness::Ready,
            CodecExtradata::None,
        )])
        .expect_err("missing config must be rejected");
    assert!(err.to_string().contains("H264"));

    publisher
        .update_tracks(vec![h264_track(
            TrackReadiness::PendingConfig,
            CodecExtradata::None,
        )])
        .expect("pending tracks may be announced");
}

#[test]
fn hub_backpressure_drop_droppable_first() {
    let hub = MemoryStreamHub::new();
    let publisher = hub
        .publish("mock://bp-droppable", PublisherOptions::default())
        .expect("publish");
    let mut subscriber = hub
        .subscribe(
            "mock://bp-droppable",
            subscriber_options(2, BackpressurePolicy::DropDroppableFirst),
        )
        .expect("subscribe");

    publisher.push_frame(video_frame(0, true, b"k1")).unwrap();
    publisher.push_frame(video_frame(40, false, b"d1")).unwrap();
    publisher.push_frame(video_frame(80, false, b"d2")).unwrap();
    publisher.close().unwrap();

    let first = subscriber.recv_blocking().unwrap().unwrap();
    let second = subscriber.recv_blocking().unwrap().unwrap();
    assert_eq!(first.meta.pts, Some(0), "keyframe must be retained");
    assert_eq!(second.meta.pts, Some(80), "droppable frame must be evicted");
    assert!(subscriber.recv_blocking().unwrap().is_none());
}

#[test]
fn hub_backpressure_drop_until_next_keyframe() {
    let hub = MemoryStreamHub::new();
    let publisher = hub
        .publish("mock://bp-keyframe", PublisherOptions::default())
        .expect("publish");
    let mut subscriber = hub
        .subscribe(
            "mock://bp-keyframe",
            subscriber_options(1, BackpressurePolicy::DropUntilNextKeyframe),
        )
        .expect("subscribe");

    publisher.push_frame(video_frame(0, true, b"k1")).unwrap();
    assert_eq!(
        publisher.push_frame(video_frame(40, false, b"d1")).unwrap(),
        DispatchResult::DroppedByPolicy
    );
    assert_eq!(
        publisher.push_frame(video_frame(80, false, b"d2")).unwrap(),
        DispatchResult::DroppedByPolicy
    );
    publisher.push_frame(video_frame(120, true, b"k2")).unwrap();
    publisher.close().unwrap();

    let frame = subscriber.recv_blocking().unwrap().unwrap();
    assert_eq!(frame.meta.pts, Some(120), "queue restarts at next keyframe");
    assert!(subscriber.recv_blocking().unwrap().is_none());
}

#[test]
fn hub_backpressure_disconnect_on_overflow() {
    let hub = MemoryStreamHub::new();
    let publisher = hub
        .publish("mock://bp-disconnect", PublisherOptions::default())
        .expect("publish");
    let mut subscriber = hub
        .subscribe(
            "mock://bp-disconnect",
            subscriber_options(1, BackpressurePolicy::DisconnectOnOverflow),
        )
        .expect("subscribe");

    publisher.push_frame(video_frame(0, true, b"k1")).unwrap();
    publisher.push_frame(video_frame(40, false, b"d1")).unwrap();

    let err = subscriber.recv_blocking().expect_err("overflow error");
    assert!(matches!(err, Error::Overflow(_)));
    assert_eq!(hub.subscriber_count("mock://bp-disconnect"), 0);
}

#[test]
fn hub_keyframe_requests_reach_publisher() {
    let hub = MemoryStreamHub::new();
    let publisher = hub
        .publish("mock://keyframe-req", PublisherOptions::default())
        .expect("publish");
    hub.request_keyframe("mock://keyframe-req")
        .expect("request");
    hub.request_keyframe("mock://keyframe-req")
        .expect("request");
    assert_eq!(publisher.take_keyframe_requests(), 2);
    assert_eq!(publisher.take_keyframe_requests(), 0);
}

#[test]
fn connector_rejects_unknown_scheme_and_direction() {
    let Err(err) = open_pull(
        StreamProtocol::RtspPull,
        "file:///tmp/clip",
        SubscriberOptions::default(),
    ) else {
        panic!("unknown scheme must fail");
    };
    assert!(matches!(err, Error::InvalidArgument(_)));

    let Err(err) = open_pull(
        StreamProtocol::RtmpPush,
        "mock://direction",
        SubscriberOptions::default(),
    ) else {
        panic!("push protocol cannot pull");
    };
    assert!(matches!(err, Error::InvalidArgument(_)));

    let Err(err) = open_push(
        StreamProtocol::HttpFlvPull,
        "mock://direction",
        PublisherOptions::default(),
    ) else {
        panic!("pull protocol cannot push");
    };
    assert!(matches!(err, Error::InvalidArgument(_)));
}

#[cfg(not(feature = "cheetah"))]
#[test]
fn connector_requires_cheetah_feature_for_network_schemes() {
    let Err(err) = open_pull(
        StreamProtocol::RtspPull,
        "rtsp://camera.local/stream",
        SubscriberOptions::default(),
    ) else {
        panic!("rtsp requires cheetah");
    };
    assert!(err.to_string().contains("cheetah"));

    let Err(err) = open_push(
        StreamProtocol::WebRtcPush,
        "whip://sfu.local/session",
        PublisherOptions::default(),
    ) else {
        panic!("whip requires cheetah");
    };
    assert!(err.to_string().contains("cheetah"));
}

#[test]
fn stream_elements_are_registered() {
    for (kind, inputs, outputs) in [
        ("rtsp_src", 0, 1),
        ("httpflv_src", 0, 1),
        ("rtmp_sink", 1, 0),
        ("webrtc_sink", 1, 0),
    ] {
        let descriptor = find_element(kind).unwrap_or_else(|| panic!("{kind} not registered"));
        assert_eq!(descriptor.input_ports.len(), inputs, "{kind} inputs");
        assert_eq!(descriptor.output_ports.len(), outputs, "{kind} outputs");
    }
}

#[test]
fn stream_element_parameters_are_validated_at_load_time() {
    let valid_track = h264_track(TrackReadiness::Ready, h264_extradata());
    let mut unknown_track = serde_json::to_value(&valid_track).expect("serialize track");
    unknown_track
        .as_object_mut()
        .expect("track object")
        .insert("unknown".to_string(), serde_json::json!(true));
    let invalid = [
        (
            "rtsp_src",
            serde_json::json!({ "url": "rtsp://camera/stream", "unknown": true }),
            "unknown field `unknown`",
        ),
        (
            "rtsp_src",
            serde_json::json!({ "url": "http://camera/stream.flv" }),
            "scheme `http` is not supported by the rtsp protocol",
        ),
        (
            "httpflv_src",
            serde_json::json!({ "url": "http://camera/stream.flv", "queue_capacity": 0 }),
            "field queue_capacity must be non-zero",
        ),
        (
            "httpflv_src",
            serde_json::json!({
                "url": "https://camera/stream.flv",
                "enable_video": false,
                "enable_audio": false
            }),
            "at least one of enable_video or enable_audio must be true",
        ),
        (
            "rtmp_sink",
            serde_json::json!({ "url": "rtmp://server/live", "backpressure": "x" }),
            "unknown field `backpressure`",
        ),
        (
            "webrtc_sink",
            serde_json::json!({ "url": "webrtc://server/session", "tracks": [unknown_track] }),
            "unknown field `unknown`",
        ),
        (
            "rtmp_sink",
            serde_json::json!({
                "url": "rtmp://server/live",
                "tracks": [serde_json::to_value(h264_track(
                    TrackReadiness::Ready,
                    CodecExtradata::None,
                )).expect("serialize track")]
            }),
            "missing required codec config",
        ),
    ];

    for (kind, params, expected) in invalid {
        let err = GraphSpecBuilder::new()
            .add_node(node("stream", kind, params))
            .build()
            .expect_err("invalid stream params must fail during graph loading");
        let message = err.to_string();
        assert!(message.contains("nodes[stream].params"), "{message}");
        assert!(message.contains(expected), "{message}");
    }
}

#[test]
fn pull_element_rejects_not_ready_tracks() {
    let hub = MemoryStreamHub::global();
    let publisher = hub
        .publish("mock://e2e/not-ready", PublisherOptions::default())
        .expect("publish");
    publisher
        .update_tracks(vec![h264_track(
            TrackReadiness::PendingConfig,
            CodecExtradata::None,
        )])
        .expect("tracks");

    let spec = GraphSpecBuilder::new()
        .add_node(node(
            "src",
            "rtsp_src",
            serde_json::json!({ "url": "mock://e2e/not-ready" }),
        ))
        .build()
        .expect("spec");
    let err = Graph::new(spec)
        .expect("graph")
        .run()
        .expect_err("not-ready track must fail");
    assert!(err.to_string().contains("not ready"));
}

#[test]
fn end_to_end_pull_push_pipeline() {
    let hub = MemoryStreamHub::global();
    let in_url = "mock://e2e/in";
    let out_url = "mock://e2e/out";

    let track = h264_track(TrackReadiness::Ready, h264_extradata());
    let publisher = hub
        .publish(in_url, PublisherOptions::default())
        .expect("publish");
    publisher
        .update_tracks(vec![track.clone()])
        .expect("tracks");

    let mut out_subscriber = hub
        .subscribe(
            out_url,
            subscriber_options(16, BackpressurePolicy::DropDroppableFirst),
        )
        .expect("subscribe output");

    let feeder = thread::spawn(move || {
        while MemoryStreamHub::global().subscriber_count(in_url) == 0 {
            thread::sleep(Duration::from_millis(2));
        }
        publisher.push_frame(video_frame(0, true, b"key0")).unwrap();
        publisher
            .push_frame(video_frame(40, false, b"delta1"))
            .unwrap();
        publisher
            .push_frame(video_frame(80, false, b"delta2"))
            .unwrap();
        publisher.close().unwrap();
    });

    let spec = GraphSpecBuilder::new()
        .add_node(node(
            "src",
            "rtsp_src",
            serde_json::json!({ "url": in_url }),
        ))
        .add_node(node(
            "sink",
            "rtmp_sink",
            serde_json::json!({
                "url": out_url,
                "tracks": serde_json::to_value(vec![track.clone()]).unwrap(),
            }),
        ))
        .connect("src.out -> sink.in")
        .build()
        .expect("spec");
    Graph::new(spec)
        .expect("graph")
        .run()
        .expect("pipeline run");
    feeder.join().expect("feeder");

    assert_eq!(hub.tracks(out_url), vec![track]);

    let mut received = Vec::new();
    while let Some(frame) = out_subscriber.recv_blocking().expect("recv") {
        received.push(frame);
    }
    assert_eq!(received.len(), 3);
    assert_eq!(received[0].meta.pts, Some(0));
    assert_eq!(
        received[0].meta.tags.get(KEYFRAME_TAG).map(String::as_str),
        Some("true")
    );
    assert_eq!(received[1].meta.pts, Some(40));
    assert_eq!(received[2].meta.pts, Some(80));
    assert_eq!(received[2].buffer.read_bytes(), b"delta2");
}

#[test]
fn stream_push_counts_policy_drops_in_element_metrics() {
    let hub = MemoryStreamHub::global();
    let in_url = "mock://metrics/in";
    let out_url = "mock://metrics/out";
    let track = h264_track(TrackReadiness::Ready, h264_extradata());
    let publisher = hub
        .publish(in_url, PublisherOptions::default())
        .expect("publish input");
    publisher
        .update_tracks(vec![track.clone()])
        .expect("input tracks");
    let _out_subscriber = hub
        .subscribe(
            out_url,
            subscriber_options(1, BackpressurePolicy::DropUntilNextKeyframe),
        )
        .expect("subscribe output");

    let feeder = thread::spawn(move || {
        while MemoryStreamHub::global().subscriber_count(in_url) == 0 {
            thread::sleep(Duration::from_millis(2));
        }
        publisher.push_frame(video_frame(0, true, b"key")).unwrap();
        publisher
            .push_frame(video_frame(40, false, b"delta"))
            .unwrap();
        publisher.close().unwrap();
    });

    let spec = GraphSpecBuilder::new()
        .add_node(node(
            "src",
            "rtsp_src",
            serde_json::json!({ "url": in_url }),
        ))
        .add_node(node(
            "sink",
            "rtmp_sink",
            serde_json::json!({
                "url": out_url,
                "tracks": serde_json::to_value(vec![track]).unwrap(),
            }),
        ))
        .connect("src.out -> sink.in")
        .build()
        .expect("spec");
    let report = Graph::new(spec)
        .expect("graph")
        .run()
        .expect("pipeline run");
    feeder.join().expect("feeder");

    assert!(
        report
            .element_metrics
            .get("sink")
            .expect("sink metrics")
            .drop_count
            > 0
    );
}
