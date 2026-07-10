use bytes::Bytes;
use serde::{Deserialize, Serialize};

/// Media kind classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MediaKind {
    Video,
    Audio,
    Data,
    Subtitle,
}

/// Track codec identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodecId {
    H264,
    H265,
    H266,
    AV1,
    VP8,
    VP9,
    MJPEG,
    AAC,
    ADPCM,
    Opus,
    G711A,
    G711U,
    MP2,
    MP3,
    Unknown,
}

/// Rational frame-rate representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rational32 {
    pub num: u32,
    pub den: u32,
}

impl Rational32 {
    pub const fn new(num: u32, den: u32) -> Self {
        Self { num, den }
    }
}

/// AAC RTP packetization modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AacRtpPacketization {
    #[default]
    Mpeg4Generic,
    Latm,
}

/// Readiness of a stream track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackReadiness {
    NotReady,
    PendingConfig,
    Ready,
}

/// Codec extra data attached to a track.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodecExtradata {
    #[default]
    None,
    H264 {
        sps: Vec<Bytes>,
        pps: Vec<Bytes>,
        avcc: Option<Bytes>,
    },
    H265 {
        vps: Vec<Bytes>,
        sps: Vec<Bytes>,
        pps: Vec<Bytes>,
        hvcc: Option<Bytes>,
    },
    H266 {
        vps: Vec<Bytes>,
        sps: Vec<Bytes>,
        pps: Vec<Bytes>,
    },
    AAC {
        asc: Bytes,
    },
    AV1 {
        sequence_header: Option<Bytes>,
        codec_config: Option<Bytes>,
    },
    VP8 {
        config: Option<Bytes>,
    },
    VP9 {
        config: Option<Bytes>,
    },
    MP3 {
        side_info: Option<Bytes>,
    },
    Opus {
        fmtp: Option<String>,
        channel_mapping: Option<Bytes>,
    },
    Raw(Bytes),
}

/// Codec configuration requirements for a track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodecConfigRequirement {
    Required,
    Optional,
    None,
}

/// Codec configuration payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodecConfigPayload {
    H264 {
        sps: Vec<Bytes>,
        pps: Vec<Bytes>,
        avcc: Option<Bytes>,
    },
    H265 {
        vps: Vec<Bytes>,
        sps: Vec<Bytes>,
        pps: Vec<Bytes>,
        hvcc: Option<Bytes>,
    },
    H266 {
        vps: Vec<Bytes>,
        sps: Vec<Bytes>,
        pps: Vec<Bytes>,
    },
    AAC {
        asc: Bytes,
    },
    AV1 {
        sequence_header: Option<Bytes>,
        codec_config: Option<Bytes>,
    },
    VP8 {
        config: Option<Bytes>,
    },
    VP9 {
        config: Option<Bytes>,
    },
    MP3 {
        side_info: Option<Bytes>,
    },
    Opus {
        fmtp: Option<String>,
        channel_mapping: Option<Bytes>,
    },
    None,
}

/// View of codec config requirements and payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodecConfigView {
    pub requirement: CodecConfigRequirement,
    pub payload: CodecConfigPayload,
}

/// Codec configuration validation errors.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodecConfigError {
    #[error("track {track_id:?} codec {codec:?} missing required codec config: {detail}")]
    MissingRequiredConfig {
        track_id: u64,
        codec: CodecId,
        detail: &'static str,
    },
}

impl CodecId {
    /// Codec configuration requirement for streaming this codec.
    pub const fn config_requirement(self) -> CodecConfigRequirement {
        match self {
            Self::H264 | Self::H265 | Self::H266 | Self::AAC => CodecConfigRequirement::Required,
            Self::AV1 | Self::VP8 | Self::VP9 | Self::MP3 | Self::Opus => {
                CodecConfigRequirement::Optional
            }
            Self::MJPEG | Self::ADPCM | Self::G711A | Self::G711U | Self::MP2 | Self::Unknown => {
                CodecConfigRequirement::None
            }
        }
    }
}

impl CodecExtradata {
    /// Converts the extradata into a codec config payload.
    pub fn to_config_payload(&self) -> CodecConfigPayload {
        match self.clone() {
            Self::None | Self::Raw(_) => CodecConfigPayload::None,
            Self::H264 { sps, pps, avcc } => CodecConfigPayload::H264 { sps, pps, avcc },
            Self::H265 {
                vps,
                sps,
                pps,
                hvcc,
            } => CodecConfigPayload::H265 {
                vps,
                sps,
                pps,
                hvcc,
            },
            Self::H266 { vps, sps, pps } => CodecConfigPayload::H266 { vps, sps, pps },
            Self::AAC { asc } => CodecConfigPayload::AAC { asc },
            Self::AV1 {
                sequence_header,
                codec_config,
            } => CodecConfigPayload::AV1 {
                sequence_header,
                codec_config,
            },
            Self::VP8 { config } => CodecConfigPayload::VP8 { config },
            Self::VP9 { config } => CodecConfigPayload::VP9 { config },
            Self::MP3 { side_info } => CodecConfigPayload::MP3 { side_info },
            Self::Opus {
                fmtp,
                channel_mapping,
            } => CodecConfigPayload::Opus {
                fmtp,
                channel_mapping,
            },
        }
    }

    /// Returns the missing-config detail if this extradata does not satisfy the codec.
    fn missing_detail(&self, codec: CodecId) -> Option<&'static str> {
        match codec {
            CodecId::H264 => match self {
                Self::H264 { sps, pps, avcc } => {
                    if avcc.is_some() || (!sps.is_empty() && !pps.is_empty()) {
                        None
                    } else {
                        Some("H264 requires sps+pps or avcc")
                    }
                }
                _ => Some("H264 requires sps/pps extradata"),
            },
            CodecId::H265 => match self {
                Self::H265 {
                    vps,
                    sps,
                    pps,
                    hvcc,
                } => {
                    if hvcc.is_some() || (!vps.is_empty() && !sps.is_empty() && !pps.is_empty()) {
                        None
                    } else {
                        Some("H265 requires vps+sps+pps or hvcc")
                    }
                }
                _ => Some("H265 requires vps/sps/pps extradata"),
            },
            CodecId::H266 => match self {
                Self::H266 { vps, sps, pps } => {
                    if !vps.is_empty() && !sps.is_empty() && !pps.is_empty() {
                        None
                    } else {
                        Some("H266 requires vps+sps+pps")
                    }
                }
                _ => Some("H266 requires vps/sps/pps extradata"),
            },
            CodecId::AAC => match self {
                Self::AAC { asc } if !asc.is_empty() => None,
                _ => Some("AAC requires a non-empty AudioSpecificConfig"),
            },
            _ => None,
        }
    }
}

/// Track-level codec and timing metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrackInfo {
    pub track_id: u64,
    pub media_kind: MediaKind,
    pub codec: CodecId,
    pub aac_rtp_packetization: AacRtpPacketization,
    pub aac_latm_config_in_band: bool,
    pub payload_type: Option<u8>,
    pub clock_rate: u32,
    pub sample_rate: Option<u32>,
    pub channels: Option<u8>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<Rational32>,
    pub bitrate: Option<u32>,
    pub extradata: CodecExtradata,
    pub readiness: TrackReadiness,
}

impl TrackInfo {
    pub fn new(track_id: u64, media_kind: MediaKind, codec: CodecId, clock_rate: u32) -> Self {
        Self {
            track_id,
            media_kind,
            codec,
            aac_rtp_packetization: AacRtpPacketization::default(),
            aac_latm_config_in_band: false,
            payload_type: None,
            clock_rate,
            sample_rate: None,
            channels: None,
            width: None,
            height: None,
            fps: None,
            bitrate: None,
            extradata: CodecExtradata::None,
            readiness: TrackReadiness::NotReady,
        }
    }

    /// Codec config requirement and current payload for this track.
    pub fn codec_config_view(&self) -> CodecConfigView {
        CodecConfigView {
            requirement: self.codec.config_requirement(),
            payload: self.extradata.to_config_payload(),
        }
    }

    /// Validates that required codec configuration is present.
    pub fn validate_codec_config(&self) -> core::result::Result<(), CodecConfigError> {
        if self.codec.config_requirement() != CodecConfigRequirement::Required {
            return Ok(());
        }
        match self.extradata.missing_detail(self.codec) {
            None => Ok(()),
            Some(detail) => Err(CodecConfigError::MissingRequiredConfig {
                track_id: self.track_id,
                codec: self.codec,
                detail,
            }),
        }
    }

    /// A track is streamable when it is `Ready` and its codec config is complete.
    pub fn is_streamable(&self) -> bool {
        self.readiness == TrackReadiness::Ready && self.validate_codec_config().is_ok()
    }
}
