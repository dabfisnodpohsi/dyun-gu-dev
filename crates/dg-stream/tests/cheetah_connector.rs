#![cfg(feature = "cheetah")]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use dg_media::{
    MediaFrame, MediaFrameKind, MediaStreamCodec, MediaStreamFormat, MediaStreamKind,
    MediaStreamMetadata, MediaStreamTimebase,
};
use dg_stream::{
    cheetah_avframe_to_media_frame, CheetahPublisherSinkAdapter, CheetahRuntimeConnector,
    EmbeddedCheetahRuntimeConnector, Error, PublisherSink, StreamProtocol,
    TrackInfo as StreamTrackInfo,
};
use dg_stream_cheetah::cheetah_connector::{
    ConnectorBuilder, LoopbackLayer, LoopbackOptions, LoopbackTopology, Protocol,
};
use dg_stream_cheetah::cheetah_runtime_tokio::TokioRuntime;
use dg_stream_cheetah::{
    AVFrame, CodecId, FrameFlags, FrameFormat, MediaKind, SdkError, Timebase, TrackId,
    TrackInfo as CheetahTrackInfo, TrackReadiness,
};

#[derive(Clone, Default)]
struct CapturingSink {
    frame: Arc<Mutex<Option<AVFrame>>>,
}

impl dg_stream_cheetah::PublisherSink for CapturingSink {
    fn update_tracks(&self, _tracks: Vec<dg_stream_cheetah::TrackInfo>) -> Result<(), SdkError> {
        Ok(())
    }

    fn push_frame(
        &self,
        frame: Arc<AVFrame>,
    ) -> Result<dg_stream_cheetah::DispatchResult, SdkError> {
        let mut captured = self
            .frame
            .lock()
            .map_err(|_| SdkError::Internal("capture lock poisoned".to_string()))?;
        *captured = Some(frame.as_ref().clone());
        Ok(dg_stream_cheetah::DispatchResult::Accepted)
    }

    fn close(&self) -> Result<(), SdkError> {
        Ok(())
    }

    fn take_keyframe_requests(&self) -> u64 {
        0
    }
}

fn metadata_frame() -> AVFrame {
    let mut frame = AVFrame::new(
        TrackId(7),
        MediaKind::Video,
        CodecId::H264,
        FrameFormat::CanonicalH26x,
        1234,
        1200,
        Timebase::new(1, 90_000),
        Bytes::from_static(&[0, 0, 0, 1, 0x65]),
    );
    frame.flags = FrameFlags::KEY;
    frame
}

fn media_frame(payload: &[u8]) -> MediaFrame {
    MediaFrame::from_host_bytes(
        MediaFrameKind::Image,
        dg_core::DataType::U8,
        dg_core::DataFormat::Auto,
        vec![payload.len()],
        dg_core::DeviceKind::Cpu,
        payload.to_vec(),
    )
    .expect("test frame")
}

#[test]
fn cheetah_bridge_preserves_frame_metadata() {
    let source = metadata_frame();
    let converted = cheetah_avframe_to_media_frame(Arc::new(source.clone()));
    assert_eq!(converted.meta.pts, Some(1234));
    assert_eq!(converted.meta.dts, Some(1200));
    assert_eq!(
        converted.meta.stream_metadata,
        Some(MediaStreamMetadata {
            track_id: 7,
            media_kind: MediaStreamKind::Video,
            codec: MediaStreamCodec::H264,
            format: MediaStreamFormat::CanonicalH26x,
            timebase: MediaStreamTimebase::new(1, 90_000),
            keyframe: true,
        })
    );

    let capture = CapturingSink::default();
    let captured = Arc::clone(&capture.frame);
    let adapter = CheetahPublisherSinkAdapter::new(Box::new(capture));
    adapter.push_frame(Arc::new(converted)).expect("push frame");
    let pushed = captured
        .lock()
        .expect("capture lock")
        .clone()
        .expect("captured frame");
    assert_eq!(pushed.track_id, source.track_id);
    assert_eq!(pushed.media_kind, source.media_kind);
    assert_eq!(pushed.codec, source.codec);
    assert_eq!(pushed.format, source.format);
    assert_eq!(pushed.timebase, source.timebase);
    assert_eq!(pushed.pts, source.pts);
    assert_eq!(pushed.dts, source.dts);
    assert!(pushed.is_key_frame());
}

#[test]
fn cheetah_bridge_resolves_metadata_from_announced_track() {
    let capture = CapturingSink::default();
    let captured = Arc::clone(&capture.frame);
    let adapter = CheetahPublisherSinkAdapter::new(Box::new(capture));
    let mut track = StreamTrackInfo::new(
        7,
        dg_stream::MediaKind::Video,
        dg_stream::CodecId::H264,
        90_000,
    );
    track.readiness = dg_stream::TrackReadiness::Ready;
    adapter.update_tracks(vec![track]).expect("announce track");

    let mut frame = media_frame(&[0, 1, 2]);
    frame.meta.stream_id = Some("7".to_string());
    frame.meta.pts = Some(55);
    frame.meta.dts = Some(44);
    frame
        .meta
        .tags
        .insert(dg_stream::KEYFRAME_TAG.to_string(), "true".to_string());
    adapter
        .push_frame(Arc::new(frame))
        .expect("push fallback frame");

    let pushed = captured
        .lock()
        .expect("capture lock")
        .clone()
        .expect("captured frame");
    assert_eq!(pushed.track_id, TrackId(7));
    assert_eq!(pushed.media_kind, MediaKind::Video);
    assert_eq!(pushed.codec, CodecId::H264);
    assert_eq!(pushed.format, FrameFormat::CanonicalH26x);
    assert_eq!(pushed.timebase, Timebase::new(1, 90_000));
    assert_eq!(pushed.pts, 55);
    assert_eq!(pushed.dts, 44);
    assert!(pushed.is_key_frame());
}

#[test]
fn embedded_connector_routes_all_stream_protocols() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    runtime.block_on(async {
        let connector = EmbeddedCheetahRuntimeConnector::new().expect("embedded connector");
        let pull_cases = [
            (StreamProtocol::RtspPull, "rtsp://"),
            (StreamProtocol::HttpFlvPull, "http://"),
        ];
        for (protocol, url) in pull_cases {
            let error = match connector.open_pull(protocol, url, Default::default()) {
                Ok(_) => panic!("invalid endpoint unexpectedly opened"),
                Err(error) => error,
            };
            assert!(matches!(error, Error::Sdk(_)), "{error:?}");
        }

        let push_cases = [
            (StreamProtocol::RtmpPush, "rtmp://"),
            (StreamProtocol::WebRtcPush, "webrtc://"),
        ];
        for (protocol, url) in push_cases {
            let error = match connector.open_push(protocol, url, Default::default()) {
                Ok(_) => panic!("invalid endpoint unexpectedly opened"),
                Err(error) => error,
            };
            assert!(matches!(error, Error::Sdk(_)), "{error:?}");
        }
    });
}

#[test]
fn engine_only_loopback_roundtrips_h264_without_sockets() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(2)
        .build()
        .expect("test runtime");
    runtime
        .block_on(async {
            let runtime_api =
                Arc::new(TokioRuntime::new()) as Arc<dyn dg_stream_cheetah::RuntimeApi>;
            let connector = ConnectorBuilder::new(runtime_api)
                .without_default_modules()
                .build()?;
            connector.start().await?;

            let mut track =
                CheetahTrackInfo::new(TrackId(0), MediaKind::Video, CodecId::H264, 90_000);
            track.readiness = TrackReadiness::Ready;
            track.extradata = dg_stream_cheetah::CodecExtradata::H264 {
                sps: vec![Bytes::from_static(&[0x67, 0x42, 0x00, 0x1f])],
                pps: vec![Bytes::from_static(&[0x68, 0xce, 0x3c, 0x80])],
                avcc: None,
            };

            let options = LoopbackOptions {
                stream_name: "dg_stream_engine_only".to_string(),
                topology: LoopbackTopology::SameProtocol {
                    protocol: Protocol::Rtmp,
                },
                preferred_layer: LoopbackLayer::EngineOnlyBypassWire,
                tracks: vec![track],
                ..Default::default()
            };

            let mut pair = connector.open_in_memory_loopback(options).await?;
            assert_eq!(pair.layer, LoopbackLayer::EngineOnlyBypassWire);
            pair.publisher.wait_ready().await?;

            let mut frame = AVFrame::new(
                TrackId(0),
                MediaKind::Video,
                CodecId::H264,
                FrameFormat::CanonicalH26x,
                0,
                0,
                Timebase::new(1, 1_000),
                Bytes::from_static(&[
                    0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00, 0x2f, 0xff, 0xff, 0x00, 0x04,
                    0x00, 0x00, 0x04, 0x01,
                ]),
            );
            frame.flags = FrameFlags::KEY;
            pair.publisher.push_frame(Arc::new(frame))?;

            let received = tokio::time::timeout(Duration::from_secs(5), pair.subscriber.recv())
                .await??
                .ok_or("loopback subscriber ended")?;
            assert_eq!(received.codec, CodecId::H264);
            assert_eq!(received.media_kind, MediaKind::Video);
            assert!(!received.payload.is_empty());

            pair.publisher.close()?;
            pair.subscriber.close().await?;
            connector.stop().await;
            Ok::<(), Box<dyn std::error::Error>>(())
        })
        .expect("engine-only loopback");
}
