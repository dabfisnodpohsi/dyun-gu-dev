use std::sync::Arc;

#[cfg(feature = "cheetah")]
use bytes::Bytes;

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
use dg_media::MediaFrameKind;

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
    media_frame
}

#[cfg(feature = "cheetah")]
pub fn media_frame_to_cheetah_avframe(
    frame: Arc<MediaFrame>,
    track_id: u64,
    media_kind: dg_stream_cheetah::MediaKind,
    codec: dg_stream_cheetah::CodecId,
    format: dg_stream_cheetah::FrameFormat,
) -> dg_stream_cheetah::AVFrame {
    let frame = match Arc::try_unwrap(frame) {
        Ok(frame) => frame,
        Err(frame) => frame.as_ref().clone(),
    };
    let payload = Bytes::from(frame.buffer.into_host_bytes());
    dg_stream_cheetah::AVFrame::new(
        dg_stream_cheetah::TrackId(u32::try_from(track_id).unwrap_or(u32::MAX)),
        media_kind,
        codec,
        format,
        frame.meta.pts.unwrap_or_default(),
        frame.meta.dts.unwrap_or_default(),
        dg_stream_cheetah::Timebase::new(1, 1),
        payload,
    )
}

#[cfg(feature = "cheetah")]
pub struct CheetahPublisherSinkAdapter {
    inner: Box<dyn dg_stream_cheetah::PublisherSink>,
}

#[cfg(feature = "cheetah")]
impl CheetahPublisherSinkAdapter {
    pub fn new(inner: Box<dyn dg_stream_cheetah::PublisherSink>) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "cheetah")]
impl PublisherSink for CheetahPublisherSinkAdapter {
    fn update_tracks(&self, tracks: Vec<TrackInfo>) -> Result<()> {
        let tracks = tracks
            .into_iter()
            .map(|track| media_frame_to_cheetah_track_info(&track))
            .collect();
        self.inner
            .update_tracks(tracks)
            .map_err(|err| Error::Sdk(err.to_string()))
    }

    fn push_frame(&self, frame: Arc<MediaFrame>) -> Result<DispatchResult> {
        let avframe = media_frame_to_cheetah_avframe(
            frame,
            0,
            dg_stream_cheetah::MediaKind::Data,
            dg_stream_cheetah::CodecId::Unknown,
            dg_stream_cheetah::FrameFormat::Unknown,
        );
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
