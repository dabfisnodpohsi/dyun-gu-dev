/// Media kind carried by a streaming frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MediaStreamKind {
    Video,
    Audio,
    Data,
    Subtitle,
}

/// Codec identity carried by a streaming frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MediaStreamCodec {
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

/// Canonical payload format carried by a streaming frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MediaStreamFormat {
    CanonicalH26x,
    CanonicalAv1Obu,
    CanonicalVp8Frame,
    CanonicalVp9Frame,
    MjpegFrame,
    AacRaw,
    AdpcmPacket,
    OpusPacket,
    G711Packet,
    Mp2Frame,
    Mp3Frame,
    DataPacket,
    Unknown,
}

/// Rational timebase carried by a streaming frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MediaStreamTimebase {
    pub num: u32,
    pub den: u32,
}

impl MediaStreamTimebase {
    pub const fn new(num: u32, den: u32) -> Self {
        Self { num, den }
    }
}

/// Per-frame metadata needed to preserve a streaming frame across adapters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MediaStreamMetadata {
    pub track_id: u64,
    pub media_kind: MediaStreamKind,
    pub codec: MediaStreamCodec,
    pub format: MediaStreamFormat,
    pub timebase: MediaStreamTimebase,
    pub keyframe: bool,
}

#[cfg(test)]
mod tests {
    use super::{
        MediaStreamCodec, MediaStreamFormat, MediaStreamKind, MediaStreamMetadata,
        MediaStreamTimebase,
    };

    #[test]
    fn metadata_types_represent_stream_frame_identity() {
        let metadata = MediaStreamMetadata {
            track_id: 7,
            media_kind: MediaStreamKind::Video,
            codec: MediaStreamCodec::H264,
            format: MediaStreamFormat::CanonicalH26x,
            timebase: MediaStreamTimebase::new(1, 90_000),
            keyframe: true,
        };
        assert_eq!(metadata.timebase.den, 90_000);
        assert!(metadata.keyframe);
    }

    #[test]
    fn metadata_option_is_backward_compatible() {
        let metadata = dg_media_meta_default();
        assert_eq!(metadata, None);
    }

    fn dg_media_meta_default() -> Option<MediaStreamMetadata> {
        super::super::frame::MediaFrameMeta::default().stream_metadata
    }
}
