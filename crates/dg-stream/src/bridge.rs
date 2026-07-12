use std::sync::Arc;

#[cfg(feature = "cheetah")]
use bytes::Bytes;
#[cfg(feature = "cheetah")]
use std::collections::HashMap;
#[cfg(feature = "cheetah")]
use std::sync::Mutex;

#[cfg(feature = "cheetah")]
use crate::error::{Error, Result};
#[cfg(feature = "cheetah")]
use crate::ids::SubscriberId;
#[cfg(feature = "cheetah")]
use crate::stream::{DispatchResult, PublisherSink, SubscriberSource};
#[cfg(feature = "cheetah")]
use crate::track::{
    AacRtpPacketization, CodecConfigPayload, CodecConfigRequirement, CodecExtradata, CodecId,
    MediaKind, Rational32, TrackInfo, TrackReadiness,
};
use dg_media::MediaFrame;
#[cfg(feature = "cheetah")]
use dg_media::{
    MediaFrameKind, MediaStreamCodec, MediaStreamFormat, MediaStreamKind, MediaStreamMetadata,
    MediaStreamTimebase,
};
pub fn media_frame_to_frame(frame: MediaFrame) -> Arc<MediaFrame> {
    Arc::new(frame)
}

#[cfg(feature = "cheetah")]
pub fn cheetah_track_info_to_media_frame(track: &dg_stream_cheetah::TrackInfo) -> TrackInfo {
    TrackInfo {
        track_id: u64::from(track.track_id.0),
        media_kind: track.media_kind.into(),
        codec: track.codec.into(),
        aac_rtp_packetization: track.aac_rtp_packetization.into(),
        aac_latm_config_in_band: track.aac_latm_config_in_band,
        payload_type: track.payload_type,
        clock_rate: track.clock_rate,
        sample_rate: track.sample_rate,
        channels: track.channels,
        width: track.width,
        height: track.height,
        fps: track.fps.map(Into::into),
        bitrate: track.bitrate,
        extradata: track.extradata.clone().into(),
        readiness: track.readiness.into(),
    }
}

#[cfg(feature = "cheetah")]
pub fn media_frame_to_cheetah_track_info(track: &TrackInfo) -> dg_stream_cheetah::TrackInfo {
    dg_stream_cheetah::TrackInfo {
        track_id: dg_stream_cheetah::TrackId(u32::try_from(track.track_id).unwrap_or(u32::MAX)),
        media_kind: track.media_kind.into(),
        codec: track.codec.into(),
        aac_rtp_packetization: track.aac_rtp_packetization.into(),
        aac_latm_config_in_band: track.aac_latm_config_in_band,
        payload_type: track.payload_type,
        clock_rate: track.clock_rate,
        sample_rate: track.sample_rate,
        channels: track.channels,
        width: track.width,
        height: track.height,
        fps: track.fps.map(Into::into),
        bitrate: track.bitrate,
        extradata: track.extradata.clone().into(),
        readiness: track.readiness.into(),
    }
}

#[cfg(feature = "cheetah")]
impl From<dg_stream_cheetah::DispatchResult> for DispatchResult {
    fn from(value: dg_stream_cheetah::DispatchResult) -> Self {
        match value {
            dg_stream_cheetah::DispatchResult::Accepted => Self::Accepted,
            dg_stream_cheetah::DispatchResult::DroppedByPolicy => Self::DroppedByPolicy,
            dg_stream_cheetah::DispatchResult::RejectedClosed => Self::RejectedClosed,
        }
    }
}

#[cfg(feature = "cheetah")]
pub fn cheetah_avframe_to_media_frame(frame: Arc<dg_stream_cheetah::AVFrame>) -> MediaFrame {
    let frame = frame.as_ref();
    let bytes = frame.payload.clone().to_vec();
    let kind = match frame.media_kind {
        dg_stream_cheetah::MediaKind::Video => MediaFrameKind::Image,
        dg_stream_cheetah::MediaKind::Audio
        | dg_stream_cheetah::MediaKind::Data
        | dg_stream_cheetah::MediaKind::Subtitle => MediaFrameKind::Tensor,
    };
    let mut media_frame = MediaFrame::from_host_bytes(
        kind,
        dg_core::DataType::U8,
        dg_core::DataFormat::Auto,
        Vec::new(),
        dg_core::DeviceKind::Cpu,
        bytes.clone(),
    )
    .unwrap_or_else(|_| {
        let buffer = dg_core::Buffer::allocate_host(dg_core::DeviceKind::Cpu, bytes.len());
        let _ = buffer.write_from_slice(&bytes);
        MediaFrame::new(
            kind,
            dg_core::DataType::U8,
            dg_core::DataFormat::Auto,
            Vec::new(),
            dg_core::DeviceKind::Cpu,
            dg_core::MemoryDomain::Host,
            buffer,
        )
    });
    media_frame.meta.pts = Some(frame.pts);
    media_frame.meta.dts = Some(frame.dts);
    media_frame.meta.stream_id = Some(u64::from(frame.track_id.0).to_string());
    media_frame.meta.stream_metadata = Some(cheetah_frame_metadata(frame));
    media_frame
}

#[cfg(feature = "cheetah")]
pub fn media_frame_to_cheetah_avframe(
    frame: Arc<MediaFrame>,
    metadata: MediaStreamMetadata,
) -> Result<dg_stream_cheetah::AVFrame> {
    let frame = match Arc::try_unwrap(frame) {
        Ok(frame) => frame,
        Err(frame) => frame.as_ref().clone(),
    };
    let payload = Bytes::from(frame.buffer.into_host_bytes());
    let track_id = u32::try_from(metadata.track_id).map_err(|_| {
        Error::InvalidArgument(format!(
            "stream metadata track id {} exceeds cheetah TrackId range",
            metadata.track_id
        ))
    })?;
    let mut avframe = dg_stream_cheetah::AVFrame::new(
        dg_stream_cheetah::TrackId(track_id),
        media_kind_to_cheetah(metadata.media_kind),
        codec_to_cheetah(metadata.codec),
        format_to_cheetah(metadata.format),
        frame.meta.pts.unwrap_or_default(),
        frame.meta.dts.unwrap_or_default(),
        dg_stream_cheetah::Timebase::new(metadata.timebase.num, metadata.timebase.den),
        payload,
    );
    if metadata.keyframe {
        avframe.flags.insert(dg_stream_cheetah::FrameFlags::KEY);
    }
    Ok(avframe)
}

#[cfg(feature = "cheetah")]
pub struct CheetahPublisherSinkAdapter {
    inner: Box<dyn dg_stream_cheetah::PublisherSink>,
    tracks: Mutex<HashMap<u64, TrackInfo>>,
}

#[cfg(feature = "cheetah")]
impl CheetahPublisherSinkAdapter {
    pub fn new(inner: Box<dyn dg_stream_cheetah::PublisherSink>) -> Self {
        Self {
            inner,
            tracks: Mutex::new(HashMap::new()),
        }
    }
}

#[cfg(feature = "cheetah")]
impl PublisherSink for CheetahPublisherSinkAdapter {
    fn update_tracks(&self, tracks: Vec<TrackInfo>) -> Result<()> {
        let cheetah_tracks = tracks
            .iter()
            .map(media_frame_to_cheetah_track_info)
            .collect();
        self.inner
            .update_tracks(cheetah_tracks)
            .map_err(|err| Error::Sdk(err.to_string()))?;
        let mut cached = self
            .tracks
            .lock()
            .map_err(|_| Error::Runtime("cheetah track cache lock poisoned".to_string()))?;
        cached.clear();
        for track in tracks {
            cached.insert(track.track_id, track);
        }
        Ok(())
    }

    fn push_frame(&self, frame: Arc<MediaFrame>) -> Result<DispatchResult> {
        let metadata = match frame.meta.stream_metadata {
            Some(metadata) => metadata,
            None => {
                let track_id = frame
                    .meta
                    .stream_id
                    .as_deref()
                    .ok_or_else(|| {
                        Error::InvalidArgument(
                            "stream frame has no metadata or stream_id track id".to_string(),
                        )
                    })?
                    .parse::<u64>()
                    .map_err(|_| {
                        Error::InvalidArgument(
                            "stream frame stream_id is not a valid track id".to_string(),
                        )
                    })?;
                let cached = self
                    .tracks
                    .lock()
                    .map_err(|_| Error::Runtime("cheetah track cache lock poisoned".to_string()))?;
                let track = cached.get(&track_id).ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "no announced cheetah track metadata for track id {track_id}"
                    ))
                })?;
                if track.clock_rate == 0 {
                    return Err(Error::InvalidArgument(format!(
                        "announced cheetah track {track_id} has an invalid zero clock rate"
                    )));
                }
                MediaStreamMetadata {
                    track_id,
                    media_kind: media_kind_to_stream(track.media_kind),
                    codec: codec_to_stream(track.codec),
                    format: canonical_format(track.codec),
                    timebase: MediaStreamTimebase::new(1, track.clock_rate),
                    keyframe: frame
                        .meta
                        .tags
                        .get(crate::hub::KEYFRAME_TAG)
                        .is_some_and(|value| value == "true"),
                }
            }
        };
        let avframe = media_frame_to_cheetah_avframe(frame, metadata)?;
        self.inner
            .push_frame(Arc::new(avframe))
            .map(Into::into)
            .map_err(|err| Error::Sdk(err.to_string()))
    }

    fn close(&self) -> Result<()> {
        self.inner
            .close()
            .map_err(|err| Error::Sdk(err.to_string()))
    }

    fn take_keyframe_requests(&self) -> u64 {
        self.inner.take_keyframe_requests()
    }
}

#[cfg(feature = "cheetah")]
fn cheetah_frame_metadata(frame: &dg_stream_cheetah::AVFrame) -> MediaStreamMetadata {
    MediaStreamMetadata {
        track_id: u64::from(frame.track_id.0),
        media_kind: media_kind_from_cheetah(frame.media_kind),
        codec: codec_from_cheetah(frame.codec),
        format: format_from_cheetah(frame.format),
        timebase: MediaStreamTimebase::new(frame.timebase.num, frame.timebase.den),
        keyframe: frame.is_key_frame(),
    }
}

#[cfg(feature = "cheetah")]
fn media_kind_from_cheetah(value: dg_stream_cheetah::MediaKind) -> MediaStreamKind {
    match value {
        dg_stream_cheetah::MediaKind::Video => MediaStreamKind::Video,
        dg_stream_cheetah::MediaKind::Audio => MediaStreamKind::Audio,
        dg_stream_cheetah::MediaKind::Data => MediaStreamKind::Data,
        dg_stream_cheetah::MediaKind::Subtitle => MediaStreamKind::Subtitle,
    }
}

#[cfg(feature = "cheetah")]
fn media_kind_to_stream(value: MediaKind) -> MediaStreamKind {
    match value {
        MediaKind::Video => MediaStreamKind::Video,
        MediaKind::Audio => MediaStreamKind::Audio,
        MediaKind::Data => MediaStreamKind::Data,
        MediaKind::Subtitle => MediaStreamKind::Subtitle,
    }
}

#[cfg(feature = "cheetah")]
fn media_kind_to_cheetah(value: MediaStreamKind) -> dg_stream_cheetah::MediaKind {
    match value {
        MediaStreamKind::Video => dg_stream_cheetah::MediaKind::Video,
        MediaStreamKind::Audio => dg_stream_cheetah::MediaKind::Audio,
        MediaStreamKind::Data => dg_stream_cheetah::MediaKind::Data,
        MediaStreamKind::Subtitle => dg_stream_cheetah::MediaKind::Subtitle,
    }
}

#[cfg(feature = "cheetah")]
fn codec_from_cheetah(value: dg_stream_cheetah::CodecId) -> MediaStreamCodec {
    match value {
        dg_stream_cheetah::CodecId::H264 => MediaStreamCodec::H264,
        dg_stream_cheetah::CodecId::H265 => MediaStreamCodec::H265,
        dg_stream_cheetah::CodecId::H266 => MediaStreamCodec::H266,
        dg_stream_cheetah::CodecId::AV1 => MediaStreamCodec::AV1,
        dg_stream_cheetah::CodecId::VP8 => MediaStreamCodec::VP8,
        dg_stream_cheetah::CodecId::VP9 => MediaStreamCodec::VP9,
        dg_stream_cheetah::CodecId::MJPEG => MediaStreamCodec::MJPEG,
        dg_stream_cheetah::CodecId::AAC => MediaStreamCodec::AAC,
        dg_stream_cheetah::CodecId::ADPCM => MediaStreamCodec::ADPCM,
        dg_stream_cheetah::CodecId::Opus => MediaStreamCodec::Opus,
        dg_stream_cheetah::CodecId::G711A => MediaStreamCodec::G711A,
        dg_stream_cheetah::CodecId::G711U => MediaStreamCodec::G711U,
        dg_stream_cheetah::CodecId::MP2 => MediaStreamCodec::MP2,
        dg_stream_cheetah::CodecId::MP3 => MediaStreamCodec::MP3,
        dg_stream_cheetah::CodecId::Unknown => MediaStreamCodec::Unknown,
    }
}

#[cfg(feature = "cheetah")]
fn codec_to_stream(value: CodecId) -> MediaStreamCodec {
    match value {
        CodecId::H264 => MediaStreamCodec::H264,
        CodecId::H265 => MediaStreamCodec::H265,
        CodecId::H266 => MediaStreamCodec::H266,
        CodecId::AV1 => MediaStreamCodec::AV1,
        CodecId::VP8 => MediaStreamCodec::VP8,
        CodecId::VP9 => MediaStreamCodec::VP9,
        CodecId::MJPEG => MediaStreamCodec::MJPEG,
        CodecId::AAC => MediaStreamCodec::AAC,
        CodecId::ADPCM => MediaStreamCodec::ADPCM,
        CodecId::Opus => MediaStreamCodec::Opus,
        CodecId::G711A => MediaStreamCodec::G711A,
        CodecId::G711U => MediaStreamCodec::G711U,
        CodecId::MP2 => MediaStreamCodec::MP2,
        CodecId::MP3 => MediaStreamCodec::MP3,
        CodecId::Unknown => MediaStreamCodec::Unknown,
    }
}

#[cfg(feature = "cheetah")]
fn codec_to_cheetah(value: MediaStreamCodec) -> dg_stream_cheetah::CodecId {
    match value {
        MediaStreamCodec::H264 => dg_stream_cheetah::CodecId::H264,
        MediaStreamCodec::H265 => dg_stream_cheetah::CodecId::H265,
        MediaStreamCodec::H266 => dg_stream_cheetah::CodecId::H266,
        MediaStreamCodec::AV1 => dg_stream_cheetah::CodecId::AV1,
        MediaStreamCodec::VP8 => dg_stream_cheetah::CodecId::VP8,
        MediaStreamCodec::VP9 => dg_stream_cheetah::CodecId::VP9,
        MediaStreamCodec::MJPEG => dg_stream_cheetah::CodecId::MJPEG,
        MediaStreamCodec::AAC => dg_stream_cheetah::CodecId::AAC,
        MediaStreamCodec::ADPCM => dg_stream_cheetah::CodecId::ADPCM,
        MediaStreamCodec::Opus => dg_stream_cheetah::CodecId::Opus,
        MediaStreamCodec::G711A => dg_stream_cheetah::CodecId::G711A,
        MediaStreamCodec::G711U => dg_stream_cheetah::CodecId::G711U,
        MediaStreamCodec::MP2 => dg_stream_cheetah::CodecId::MP2,
        MediaStreamCodec::MP3 => dg_stream_cheetah::CodecId::MP3,
        MediaStreamCodec::Unknown => dg_stream_cheetah::CodecId::Unknown,
    }
}

#[cfg(feature = "cheetah")]
fn format_from_cheetah(value: dg_stream_cheetah::FrameFormat) -> MediaStreamFormat {
    match value {
        dg_stream_cheetah::FrameFormat::CanonicalH26x => MediaStreamFormat::CanonicalH26x,
        dg_stream_cheetah::FrameFormat::CanonicalAv1Obu => MediaStreamFormat::CanonicalAv1Obu,
        dg_stream_cheetah::FrameFormat::CanonicalVp8Frame => MediaStreamFormat::CanonicalVp8Frame,
        dg_stream_cheetah::FrameFormat::CanonicalVp9Frame => MediaStreamFormat::CanonicalVp9Frame,
        dg_stream_cheetah::FrameFormat::MjpegFrame => MediaStreamFormat::MjpegFrame,
        dg_stream_cheetah::FrameFormat::AacRaw => MediaStreamFormat::AacRaw,
        dg_stream_cheetah::FrameFormat::AdpcmPacket => MediaStreamFormat::AdpcmPacket,
        dg_stream_cheetah::FrameFormat::OpusPacket => MediaStreamFormat::OpusPacket,
        dg_stream_cheetah::FrameFormat::G711Packet => MediaStreamFormat::G711Packet,
        dg_stream_cheetah::FrameFormat::Mp2Frame => MediaStreamFormat::Mp2Frame,
        dg_stream_cheetah::FrameFormat::Mp3Frame => MediaStreamFormat::Mp3Frame,
        dg_stream_cheetah::FrameFormat::DataPacket => MediaStreamFormat::DataPacket,
        dg_stream_cheetah::FrameFormat::Unknown => MediaStreamFormat::Unknown,
    }
}

#[cfg(feature = "cheetah")]
fn format_to_cheetah(value: MediaStreamFormat) -> dg_stream_cheetah::FrameFormat {
    match value {
        MediaStreamFormat::CanonicalH26x => dg_stream_cheetah::FrameFormat::CanonicalH26x,
        MediaStreamFormat::CanonicalAv1Obu => dg_stream_cheetah::FrameFormat::CanonicalAv1Obu,
        MediaStreamFormat::CanonicalVp8Frame => dg_stream_cheetah::FrameFormat::CanonicalVp8Frame,
        MediaStreamFormat::CanonicalVp9Frame => dg_stream_cheetah::FrameFormat::CanonicalVp9Frame,
        MediaStreamFormat::MjpegFrame => dg_stream_cheetah::FrameFormat::MjpegFrame,
        MediaStreamFormat::AacRaw => dg_stream_cheetah::FrameFormat::AacRaw,
        MediaStreamFormat::AdpcmPacket => dg_stream_cheetah::FrameFormat::AdpcmPacket,
        MediaStreamFormat::OpusPacket => dg_stream_cheetah::FrameFormat::OpusPacket,
        MediaStreamFormat::G711Packet => dg_stream_cheetah::FrameFormat::G711Packet,
        MediaStreamFormat::Mp2Frame => dg_stream_cheetah::FrameFormat::Mp2Frame,
        MediaStreamFormat::Mp3Frame => dg_stream_cheetah::FrameFormat::Mp3Frame,
        MediaStreamFormat::DataPacket => dg_stream_cheetah::FrameFormat::DataPacket,
        MediaStreamFormat::Unknown => dg_stream_cheetah::FrameFormat::Unknown,
    }
}

#[cfg(feature = "cheetah")]
fn canonical_format(codec: CodecId) -> MediaStreamFormat {
    match codec {
        CodecId::H264 | CodecId::H265 | CodecId::H266 => MediaStreamFormat::CanonicalH26x,
        CodecId::AV1 => MediaStreamFormat::CanonicalAv1Obu,
        CodecId::VP8 => MediaStreamFormat::CanonicalVp8Frame,
        CodecId::VP9 => MediaStreamFormat::CanonicalVp9Frame,
        CodecId::MJPEG => MediaStreamFormat::MjpegFrame,
        CodecId::AAC => MediaStreamFormat::AacRaw,
        CodecId::ADPCM => MediaStreamFormat::AdpcmPacket,
        CodecId::Opus => MediaStreamFormat::OpusPacket,
        CodecId::G711A | CodecId::G711U => MediaStreamFormat::G711Packet,
        CodecId::MP2 => MediaStreamFormat::Mp2Frame,
        CodecId::MP3 => MediaStreamFormat::Mp3Frame,
        CodecId::Unknown => MediaStreamFormat::Unknown,
    }
}

#[cfg(feature = "cheetah")]
pub struct CheetahSubscriberSourceAdapter {
    inner: Box<dyn dg_stream_cheetah::SubscriberSource>,
}

#[cfg(feature = "cheetah")]
impl CheetahSubscriberSourceAdapter {
    pub fn new(inner: Box<dyn dg_stream_cheetah::SubscriberSource>) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "cheetah")]
#[async_trait::async_trait]
impl SubscriberSource for CheetahSubscriberSourceAdapter {
    async fn recv(&mut self) -> Result<Option<Arc<MediaFrame>>> {
        let next = self
            .inner
            .recv()
            .await
            .map_err(|err| Error::Sdk(err.to_string()))?;
        Ok(next.map(|frame| Arc::new(cheetah_avframe_to_media_frame(frame))))
    }

    async fn close(&mut self) -> Result<()> {
        self.inner
            .close()
            .await
            .map_err(|err| Error::Sdk(err.to_string()))
    }

    fn id(&self) -> SubscriberId {
        SubscriberId(self.inner.id().0)
    }
}

#[cfg(feature = "cheetah")]
impl From<dg_stream_cheetah::MediaKind> for MediaKind {
    fn from(value: dg_stream_cheetah::MediaKind) -> Self {
        match value {
            dg_stream_cheetah::MediaKind::Video => Self::Video,
            dg_stream_cheetah::MediaKind::Audio => Self::Audio,
            dg_stream_cheetah::MediaKind::Data => Self::Data,
            dg_stream_cheetah::MediaKind::Subtitle => Self::Subtitle,
        }
    }
}

#[cfg(feature = "cheetah")]
impl From<MediaKind> for dg_stream_cheetah::MediaKind {
    fn from(value: MediaKind) -> Self {
        match value {
            MediaKind::Video => Self::Video,
            MediaKind::Audio => Self::Audio,
            MediaKind::Data => Self::Data,
            MediaKind::Subtitle => Self::Subtitle,
        }
    }
}

#[cfg(feature = "cheetah")]
impl From<dg_stream_cheetah::CodecId> for CodecId {
    fn from(value: dg_stream_cheetah::CodecId) -> Self {
        match value {
            dg_stream_cheetah::CodecId::H264 => Self::H264,
            dg_stream_cheetah::CodecId::H265 => Self::H265,
            dg_stream_cheetah::CodecId::H266 => Self::H266,
            dg_stream_cheetah::CodecId::AV1 => Self::AV1,
            dg_stream_cheetah::CodecId::VP8 => Self::VP8,
            dg_stream_cheetah::CodecId::VP9 => Self::VP9,
            dg_stream_cheetah::CodecId::MJPEG => Self::MJPEG,
            dg_stream_cheetah::CodecId::AAC => Self::AAC,
            dg_stream_cheetah::CodecId::ADPCM => Self::ADPCM,
            dg_stream_cheetah::CodecId::Opus => Self::Opus,
            dg_stream_cheetah::CodecId::G711A => Self::G711A,
            dg_stream_cheetah::CodecId::G711U => Self::G711U,
            dg_stream_cheetah::CodecId::MP2 => Self::MP2,
            dg_stream_cheetah::CodecId::MP3 => Self::MP3,
            dg_stream_cheetah::CodecId::Unknown => Self::Unknown,
        }
    }
}

#[cfg(feature = "cheetah")]
impl From<CodecId> for dg_stream_cheetah::CodecId {
    fn from(value: CodecId) -> Self {
        match value {
            CodecId::H264 => Self::H264,
            CodecId::H265 => Self::H265,
            CodecId::H266 => Self::H266,
            CodecId::AV1 => Self::AV1,
            CodecId::VP8 => Self::VP8,
            CodecId::VP9 => Self::VP9,
            CodecId::MJPEG => Self::MJPEG,
            CodecId::AAC => Self::AAC,
            CodecId::ADPCM => Self::ADPCM,
            CodecId::Opus => Self::Opus,
            CodecId::G711A => Self::G711A,
            CodecId::G711U => Self::G711U,
            CodecId::MP2 => Self::MP2,
            CodecId::MP3 => Self::MP3,
            CodecId::Unknown => Self::Unknown,
        }
    }
}

#[cfg(feature = "cheetah")]
impl From<dg_stream_cheetah::AacRtpPacketization> for AacRtpPacketization {
    fn from(value: dg_stream_cheetah::AacRtpPacketization) -> Self {
        match value {
            dg_stream_cheetah::AacRtpPacketization::Mpeg4Generic => Self::Mpeg4Generic,
            dg_stream_cheetah::AacRtpPacketization::Latm => Self::Latm,
        }
    }
}

#[cfg(feature = "cheetah")]
impl From<AacRtpPacketization> for dg_stream_cheetah::AacRtpPacketization {
    fn from(value: AacRtpPacketization) -> Self {
        match value {
            AacRtpPacketization::Mpeg4Generic => Self::Mpeg4Generic,
            AacRtpPacketization::Latm => Self::Latm,
        }
    }
}

#[cfg(feature = "cheetah")]
impl From<dg_stream_cheetah::TrackReadiness> for TrackReadiness {
    fn from(value: dg_stream_cheetah::TrackReadiness) -> Self {
        match value {
            dg_stream_cheetah::TrackReadiness::NotReady => Self::NotReady,
            dg_stream_cheetah::TrackReadiness::PendingConfig => Self::PendingConfig,
            dg_stream_cheetah::TrackReadiness::Ready => Self::Ready,
        }
    }
}

#[cfg(feature = "cheetah")]
impl From<TrackReadiness> for dg_stream_cheetah::TrackReadiness {
    fn from(value: TrackReadiness) -> Self {
        match value {
            TrackReadiness::NotReady => Self::NotReady,
            TrackReadiness::PendingConfig => Self::PendingConfig,
            TrackReadiness::Ready => Self::Ready,
        }
    }
}

#[cfg(feature = "cheetah")]
impl From<dg_stream_cheetah::Rational32> for Rational32 {
    fn from(value: dg_stream_cheetah::Rational32) -> Self {
        Self {
            num: value.num,
            den: value.den,
        }
    }
}

#[cfg(feature = "cheetah")]
impl From<Rational32> for dg_stream_cheetah::Rational32 {
    fn from(value: Rational32) -> Self {
        Self::new(value.num, value.den)
    }
}

#[cfg(feature = "cheetah")]
impl From<dg_stream_cheetah::CodecExtradata> for CodecExtradata {
    fn from(value: dg_stream_cheetah::CodecExtradata) -> Self {
        match value {
            dg_stream_cheetah::CodecExtradata::None => Self::None,
            dg_stream_cheetah::CodecExtradata::H264 { sps, pps, avcc } => {
                Self::H264 { sps, pps, avcc }
            }
            dg_stream_cheetah::CodecExtradata::H265 {
                vps,
                sps,
                pps,
                hvcc,
            } => Self::H265 {
                vps,
                sps,
                pps,
                hvcc,
            },
            dg_stream_cheetah::CodecExtradata::H266 { vps, sps, pps } => {
                Self::H266 { vps, sps, pps }
            }
            dg_stream_cheetah::CodecExtradata::AAC { asc } => Self::AAC { asc },
            dg_stream_cheetah::CodecExtradata::AV1 {
                sequence_header,
                codec_config,
            } => Self::AV1 {
                sequence_header,
                codec_config,
            },
            dg_stream_cheetah::CodecExtradata::VP8 { config } => Self::VP8 { config },
            dg_stream_cheetah::CodecExtradata::VP9 { config } => Self::VP9 { config },
            dg_stream_cheetah::CodecExtradata::MP3 { side_info } => Self::MP3 { side_info },
            dg_stream_cheetah::CodecExtradata::Opus {
                fmtp,
                channel_mapping,
            } => Self::Opus {
                fmtp,
                channel_mapping,
            },
            dg_stream_cheetah::CodecExtradata::Raw(bytes) => Self::Raw(bytes),
        }
    }
}

#[cfg(feature = "cheetah")]
impl From<CodecExtradata> for dg_stream_cheetah::CodecExtradata {
    fn from(value: CodecExtradata) -> Self {
        match value {
            CodecExtradata::None => Self::None,
            CodecExtradata::H264 { sps, pps, avcc } => Self::H264 { sps, pps, avcc },
            CodecExtradata::H265 {
                vps,
                sps,
                pps,
                hvcc,
            } => Self::H265 {
                vps,
                sps,
                pps,
                hvcc,
            },
            CodecExtradata::H266 { vps, sps, pps } => Self::H266 { vps, sps, pps },
            CodecExtradata::AAC { asc } => Self::AAC { asc },
            CodecExtradata::AV1 {
                sequence_header,
                codec_config,
            } => Self::AV1 {
                sequence_header,
                codec_config,
            },
            CodecExtradata::VP8 { config } => Self::VP8 { config },
            CodecExtradata::VP9 { config } => Self::VP9 { config },
            CodecExtradata::MP3 { side_info } => Self::MP3 { side_info },
            CodecExtradata::Opus {
                fmtp,
                channel_mapping,
            } => Self::Opus {
                fmtp,
                channel_mapping,
            },
            CodecExtradata::Raw(bytes) => Self::Raw(bytes),
        }
    }
}

#[cfg(feature = "cheetah")]
impl From<dg_stream_cheetah::CodecConfigRequirement> for CodecConfigRequirement {
    fn from(value: dg_stream_cheetah::CodecConfigRequirement) -> Self {
        match value {
            dg_stream_cheetah::CodecConfigRequirement::Required => Self::Required,
            dg_stream_cheetah::CodecConfigRequirement::Optional => Self::Optional,
            dg_stream_cheetah::CodecConfigRequirement::None => Self::None,
        }
    }
}

#[cfg(feature = "cheetah")]
impl From<CodecConfigRequirement> for dg_stream_cheetah::CodecConfigRequirement {
    fn from(value: CodecConfigRequirement) -> Self {
        match value {
            CodecConfigRequirement::Required => Self::Required,
            CodecConfigRequirement::Optional => Self::Optional,
            CodecConfigRequirement::None => Self::None,
        }
    }
}

#[cfg(feature = "cheetah")]
impl From<CodecConfigPayload> for dg_stream_cheetah::CodecConfigPayload {
    fn from(value: CodecConfigPayload) -> Self {
        match value {
            CodecConfigPayload::H264 { sps, pps, avcc } => Self::H264 { sps, pps, avcc },
            CodecConfigPayload::H265 {
                vps,
                sps,
                pps,
                hvcc,
            } => Self::H265 {
                vps,
                sps,
                pps,
                hvcc,
            },
            CodecConfigPayload::H266 { vps, sps, pps } => Self::H266 { vps, sps, pps },
            CodecConfigPayload::AAC { asc } => Self::AAC { asc },
            CodecConfigPayload::AV1 {
                sequence_header,
                codec_config,
            } => Self::AV1 {
                sequence_header,
                codec_config,
            },
            CodecConfigPayload::VP8 { config } => Self::VP8 { config },
            CodecConfigPayload::VP9 { config } => Self::VP9 { config },
            CodecConfigPayload::MP3 { side_info } => Self::MP3 { side_info },
            CodecConfigPayload::Opus {
                fmtp,
                channel_mapping,
            } => Self::Opus {
                fmtp,
                channel_mapping,
            },
            CodecConfigPayload::None => Self::None,
        }
    }
}

#[cfg(feature = "cheetah")]
impl From<dg_stream_cheetah::CodecConfigPayload> for CodecConfigPayload {
    fn from(value: dg_stream_cheetah::CodecConfigPayload) -> Self {
        match value {
            dg_stream_cheetah::CodecConfigPayload::H264 { sps, pps, avcc } => {
                Self::H264 { sps, pps, avcc }
            }
            dg_stream_cheetah::CodecConfigPayload::H265 {
                vps,
                sps,
                pps,
                hvcc,
            } => Self::H265 {
                vps,
                sps,
                pps,
                hvcc,
            },
            dg_stream_cheetah::CodecConfigPayload::H266 { vps, sps, pps } => {
                Self::H266 { vps, sps, pps }
            }
            dg_stream_cheetah::CodecConfigPayload::AAC { asc } => Self::AAC { asc },
            dg_stream_cheetah::CodecConfigPayload::AV1 {
                sequence_header,
                codec_config,
            } => Self::AV1 {
                sequence_header,
                codec_config,
            },
            dg_stream_cheetah::CodecConfigPayload::VP8 { config } => Self::VP8 { config },
            dg_stream_cheetah::CodecConfigPayload::VP9 { config } => Self::VP9 { config },
            dg_stream_cheetah::CodecConfigPayload::MP3 { side_info } => Self::MP3 { side_info },
            dg_stream_cheetah::CodecConfigPayload::Opus {
                fmtp,
                channel_mapping,
            } => Self::Opus {
                fmtp,
                channel_mapping,
            },
            dg_stream_cheetah::CodecConfigPayload::None => Self::None,
        }
    }
}
