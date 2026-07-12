use dg_core::{Error, Result};

use crate::bridge::{
    avcodec_image_to_media_frame, avcodec_image_to_media_frame_with_processor,
    avcodec_packet_to_media_frame, media_frame_to_avcodec_image, media_frame_to_avcodec_packet,
};
use crate::MediaFrame;

use dg_media_avcodec::{
    AvError, AvErrorContext, AvErrorKind, BitstreamFormat, CodecId, Decoder, DecoderConfig,
    Encoder, EncoderConfig, ImageOp, ImageProcessRequest, ImageProcessor, ImageProcessorConfig,
    MemoryDomain, Poll, Registry, TimeBase,
};

pub fn registry() -> Registry {
    dg_media_avcodec::default_registry_builder().build()
}

pub fn codec_from_name(name: Option<&str>) -> Result<CodecId> {
    match name.unwrap_or("jpeg").to_ascii_lowercase().as_str() {
        "jpeg" => Ok(CodecId::Jpeg),
        "mjpeg" => Ok(CodecId::Mjpeg),
        "h264" => Ok(CodecId::H264),
        other => Err(Error::Config(format!(
            "codec must be one of `jpeg`, `mjpeg`, or `h264`, got `{other}`"
        ))),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HwPreference {
    Auto,
    Rockchip,
    Nvidia,
    Intel,
    Amd,
    Software,
}

impl HwPreference {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("auto").to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "rk" | "rockchip" | "rknn" | "rknpu" => Ok(Self::Rockchip),
            "nv" | "nvidia" | "cuda" => Ok(Self::Nvidia),
            "intel" | "vaapi" => Ok(Self::Intel),
            "amd" | "amf" => Ok(Self::Amd),
            "sw" | "software" | "cpu" | "none" => Ok(Self::Software),
            other => Err(Error::Config(format!(
                "hw must be one of `auto`, `rk`, `rockchip`, `rknn`, `rknpu`, `nv`, `nvidia`, `cuda`, `intel`, `vaapi`, `amd`, `amf`, `sw`, `software`, `cpu`, or `none`, got `{other}`"
            ))),
        }
    }
}

fn backend_candidates(codec: CodecId, hw: HwPreference, encode: bool) -> Vec<&'static str> {
    if matches!(codec, CodecId::Jpeg | CodecId::Mjpeg) {
        return if encode {
            vec!["jpeg"]
        } else {
            vec!["jpeg", "zune"]
        };
    }

    let hardware = match hw {
        HwPreference::Auto => vec!["rkmpp", "nvcodec", "onevpl", "amf"],
        HwPreference::Rockchip => vec!["rkmpp"],
        HwPreference::Nvidia => vec!["nvcodec"],
        HwPreference::Intel => vec!["onevpl"],
        HwPreference::Amd => vec!["amf"],
        HwPreference::Software => Vec::new(),
    };
    let software = if encode {
        ["ffmpeg", "x264", "openh264"].as_slice()
    } else {
        ["ffmpeg", "openh264"].as_slice()
    };
    hardware
        .into_iter()
        .chain(software.iter().copied())
        .collect()
}

fn no_backend_error(
    codec: CodecId,
    hw: HwPreference,
    candidates: &[&'static str],
    attempts: &[String],
) -> Error {
    Error::Media(format!(
        "no backend available for codec {codec:?} with hardware preference {hw:?}; attempted [{}]; enable one of cargo features: {}",
        attempts.join("; "),
        candidates
            .iter()
            .map(|candidate| match *candidate {
                "jpeg" | "zune" => "`avcodec` (jpeg/zune)".to_string(),
                other => format!("`codec-{other}`"),
            })
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

fn is_skippable_selection_error(error: &AvError) -> bool {
    matches!(
        error.kind(),
        AvErrorKind::Unsupported | AvErrorKind::SelectionFailed
    )
}

fn create_decoder(codec: CodecId, hw: HwPreference) -> Result<Box<dyn Decoder>> {
    let candidates = backend_candidates(codec, hw, false);
    let mut attempts = Vec::new();
    let registry = registry();
    let config = DecoderConfig::new(codec, TimeBase::new(1, 25))
        .with_memory_domain(MemoryDomain::Host)
        .with_allow_staging(true);
    for candidate in &candidates {
        let hinted = config.clone().with_backend_hint(Some(candidate));
        match registry.create_decoder(&hinted) {
            Ok(decoder) => return Ok(decoder),
            Err(error) if is_skippable_selection_error(&error) => {
                attempts.push(format!("{candidate}: {}", map_av_error(error)));
            }
            Err(error) => return Err(map_av_error(error)),
        }
    }
    Err(no_backend_error(codec, hw, &candidates, &attempts))
}

fn create_encoder(
    codec: CodecId,
    hw: HwPreference,
    image: &dg_media_avcodec::Image,
) -> Result<Box<dyn Encoder>> {
    let candidates = backend_candidates(codec, hw, true);
    let mut attempts = Vec::new();
    let registry = registry();
    let config = EncoderConfig::new(
        codec,
        image.coded_width,
        image.coded_height,
        image.format,
        TimeBase::new(1, 25),
        1,
    )
    .with_memory_domain(MemoryDomain::Host)
    .with_allow_staging(true);
    for candidate in &candidates {
        let hinted = config.clone().with_backend_hint(Some(candidate));
        match registry.create_encoder(&hinted) {
            Ok(encoder) => return Ok(encoder),
            Err(error) if is_skippable_selection_error(&error) => {
                attempts.push(format!("{candidate}: {}", map_av_error(error)));
            }
            Err(error) => return Err(map_av_error(error)),
        }
    }
    Err(no_backend_error(codec, hw, &candidates, &attempts))
}

fn bitstream_format(codec: CodecId) -> BitstreamFormat {
    match codec {
        CodecId::H264 => BitstreamFormat::H264AnnexB,
        CodecId::Jpeg | CodecId::Mjpeg => BitstreamFormat::JpegInterchange,
        _ => BitstreamFormat::JpegInterchange,
    }
}

pub fn map_av_error(error: AvError) -> Error {
    match error {
        AvError::InvalidArgument => Error::Media("avcodec: invalid argument".to_string()),
        AvError::Unsupported => Error::Media("avcodec: unsupported operation".to_string()),
        AvError::Again => Error::Media("avcodec: operation needs polling".to_string()),
        AvError::EndOfStream => Error::Media("avcodec: end of stream".to_string()),
        AvError::BufferDomainMismatch => {
            Error::Media("avcodec: buffer memory domain mismatch".to_string())
        }
        AvError::NotInitialized => Error::Media("avcodec: not initialized".to_string()),
        AvError::QueueFull => Error::Media("avcodec: queue full".to_string()),
        AvError::BackendFailure => Error::Media("avcodec: backend failure".to_string()),
        AvError::BackendMessage(message) => Error::Media(format!("avcodec backend: {message}")),
        AvError::InvalidState => Error::Media("avcodec: invalid state".to_string()),
        AvError::CycleDetected => Error::Media("avcodec: cycle detected".to_string()),
        AvError::DeviceLost => Error::Media("avcodec: device lost".to_string()),
        AvError::OutOfMemory => Error::Media("avcodec: out of memory".to_string()),
        AvError::Classified { kind, detail } => {
            Error::Media(format!("avcodec {kind:?}: {detail:?}"))
        }
        AvError::ExternalError(code) => Error::Media(format!("avcodec external error code {code}")),
        AvError::WithContext { error, context } => {
            append_av_error_context(map_av_error(*error), context)
        }
    }
}

fn append_av_error_context(error: Error, context: AvErrorContext) -> Error {
    let mut fields = Vec::new();
    if let Some(backend_id) = context.backend_id {
        fields.push(format!("backend={backend_id}"));
    }
    if let Some(codec) = context.codec {
        fields.push(format!("codec={codec:?}"));
    }
    if let Some(operation) = context.operation {
        fields.push(format!("operation={operation:?}"));
    }
    if let Some(frame_index) = context.frame_index {
        fields.push(format!("frame_index={frame_index}"));
    }
    if let Some(packet_index) = context.packet_index {
        fields.push(format!("packet_index={packet_index}"));
    }
    if let Some(source_format) = context.source_format {
        fields.push(format!("source_format={source_format:?}"));
    }
    if let Some(destination_format) = context.destination_format {
        fields.push(format!("destination_format={destination_format:?}"));
    }
    if let Some(width) = context.width {
        fields.push(format!("width={width}"));
    }
    if let Some(height) = context.height {
        fields.push(format!("height={height}"));
    }

    if fields.is_empty() {
        return error;
    }

    match error {
        Error::Media(message) => Error::Media(format!("{message}; context: {}", fields.join(", "))),
        other => other,
    }
}

fn is_again(error: &AvError) -> bool {
    error.kind() == AvErrorKind::Again
}

fn is_end_of_stream(error: &AvError) -> bool {
    error.kind() == AvErrorKind::EndOfStream
}

fn create_csc_processor() -> Result<Box<dyn ImageProcessor>> {
    let config = ImageProcessorConfig::new()
        .with_memory_domain(MemoryDomain::Host)
        .with_allow_staging(true)
        .with_target_op(dg_media_avcodec::ImageOpKind::Csc);
    registry()
        .create_image_processor(&config)
        .map_err(map_av_error)
}

pub struct DecodeCore {
    decoder: Box<dyn Decoder>,
    codec: CodecId,
    eos: bool,
    pending_error: Option<Error>,
    csc_processor: Option<Box<dyn ImageProcessor>>,
}

impl DecodeCore {
    pub fn new(codec: CodecId, hw: HwPreference) -> Result<Self> {
        let decoder = create_decoder(codec, hw)?;
        Ok(Self {
            decoder,
            codec,
            eos: false,
            pending_error: None,
            csc_processor: None,
        })
    }

    pub fn submit_packet(&mut self, frame: MediaFrame) -> Result<()> {
        if self.eos {
            return Err(Error::Media(
                "media_decode: packet submitted after end of stream".to_string(),
            ));
        }
        let packet =
            media_frame_to_avcodec_packet(frame, 0, self.codec, bitstream_format(self.codec))?;
        match self.decoder.submit_packet(packet) {
            Ok(()) => Ok(()),
            Err(error) if is_again(&error) || is_end_of_stream(&error) => Ok(()),
            Err(error) => Err(map_av_error(error)),
        }
    }

    pub fn submit_end_of_stream(&mut self) {
        self.eos = true;
        if let Err(error) = self.decoder.flush() {
            if !is_again(&error) && !is_end_of_stream(&error) {
                self.pending_error = Some(map_av_error(error));
            }
        }
    }

    pub fn poll(&mut self) -> Result<crate::ops::MediaPoll> {
        if let Some(error) = self.pending_error.take() {
            return Err(error);
        }
        match self.decoder.poll_frame() {
            Ok(Poll::Ready(image)) => {
                if image.format == dg_media_avcodec::ImageInfo::Yuv420p
                    && self.csc_processor.is_none()
                {
                    self.csc_processor = Some(create_csc_processor()?);
                }
                let processor = self
                    .csc_processor
                    .as_mut()
                    .map(|processor| processor.as_mut() as &mut dyn ImageProcessor);
                Ok(crate::ops::MediaPoll::Ready(
                    avcodec_image_to_media_frame_with_processor(&image, processor)?,
                ))
            }
            Ok(Poll::Pending) => Ok(crate::ops::MediaPoll::Pending),
            Ok(Poll::EndOfStream) => Ok(crate::ops::MediaPoll::EndOfStream),
            Err(error) if is_again(&error) => Ok(crate::ops::MediaPoll::Pending),
            Err(error) if is_end_of_stream(&error) => Ok(crate::ops::MediaPoll::EndOfStream),
            Err(error) => Err(map_av_error(error)),
        }
    }
}

pub struct EncodeCore {
    encoder: Option<Box<dyn Encoder>>,
    codec: CodecId,
    eos: bool,
    pending_error: Option<Error>,
    hw: HwPreference,
}

impl EncodeCore {
    pub fn new(codec: CodecId, hw: HwPreference) -> Result<Self> {
        Ok(Self {
            encoder: None,
            codec,
            eos: false,
            pending_error: None,
            hw,
        })
    }

    pub fn submit_image(&mut self, frame: MediaFrame) -> Result<()> {
        if self.eos {
            return Err(Error::Media(
                "media_encode: frame submitted after end of stream".to_string(),
            ));
        }
        let image = media_frame_to_avcodec_image(frame, 1)?;
        if self.encoder.is_none() {
            self.encoder = Some(create_encoder(self.codec, self.hw, &image)?);
        }
        let encoder = self
            .encoder
            .as_mut()
            .ok_or_else(|| Error::Media("avcodec encoder was not initialized".to_string()))?;
        match encoder.submit_frame(image) {
            Ok(()) => Ok(()),
            Err(error) if is_again(&error) || is_end_of_stream(&error) => Ok(()),
            Err(error) => Err(map_av_error(error)),
        }
    }

    pub fn submit_end_of_stream(&mut self) {
        self.eos = true;
        let Some(encoder) = self.encoder.as_mut() else {
            return;
        };
        if let Err(error) = encoder.flush() {
            if !is_again(&error) && !is_end_of_stream(&error) {
                self.pending_error = Some(map_av_error(error));
            }
        }
    }

    pub fn poll(&mut self) -> Result<crate::ops::MediaPoll> {
        if let Some(error) = self.pending_error.take() {
            return Err(error);
        }
        let Some(encoder) = self.encoder.as_mut() else {
            return Ok(crate::ops::MediaPoll::EndOfStream);
        };
        match encoder.poll_packet() {
            Ok(Poll::Ready(packet)) => Ok(crate::ops::MediaPoll::Ready(
                avcodec_packet_to_media_frame(&packet)?,
            )),
            Ok(Poll::Pending) => Ok(crate::ops::MediaPoll::Pending),
            Ok(Poll::EndOfStream) => Ok(crate::ops::MediaPoll::EndOfStream),
            Err(error) if is_again(&error) => Ok(crate::ops::MediaPoll::Pending),
            Err(error) if is_end_of_stream(&error) => Ok(crate::ops::MediaPoll::EndOfStream),
            Err(error) => Err(map_av_error(error)),
        }
    }
}

pub struct ResizeCore {
    processor: Box<dyn ImageProcessor>,
    width: u32,
    height: u32,
    eos: bool,
    pending_error: Option<Error>,
}

impl ResizeCore {
    pub fn new(width: usize, height: usize) -> Result<Self> {
        let width = u32::try_from(width)
            .map_err(|_| Error::Media("media_resize: width exceeds u32".to_string()))?;
        let height = u32::try_from(height)
            .map_err(|_| Error::Media("media_resize: height exceeds u32".to_string()))?;
        let config = ImageProcessorConfig::new()
            .with_memory_domain(MemoryDomain::Host)
            .with_target_op(dg_media_avcodec::ImageOpKind::Resize);
        let processor = registry()
            .create_image_processor(&config)
            .map_err(map_av_error)?;
        Ok(Self {
            processor,
            width,
            height,
            eos: false,
            pending_error: None,
        })
    }

    pub fn submit_image(&mut self, frame: MediaFrame) -> Result<()> {
        if self.eos {
            return Err(Error::Media(
                "media_resize: frame submitted after end of stream".to_string(),
            ));
        }
        let image = media_frame_to_avcodec_image(frame, 1)?;
        let request = ImageProcessRequest {
            src: image,
            op: ImageOp::Resize {
                width: self.width,
                height: self.height,
            },
            aux: None,
        };
        match self.processor.submit(request) {
            Ok(()) => Ok(()),
            Err(error) if is_again(&error) || is_end_of_stream(&error) => Ok(()),
            Err(error) => Err(map_av_error(error)),
        }
    }

    pub fn submit_end_of_stream(&mut self) {
        self.eos = true;
        if let Err(error) = self.processor.flush() {
            if !is_again(&error) && !is_end_of_stream(&error) {
                self.pending_error = Some(map_av_error(error));
            }
        }
    }

    pub fn poll(&mut self) -> Result<crate::ops::MediaPoll> {
        if let Some(error) = self.pending_error.take() {
            return Err(error);
        }
        match self.processor.poll_image() {
            Ok(Poll::Ready(image)) => Ok(crate::ops::MediaPoll::Ready(
                avcodec_image_to_media_frame(&image)?,
            )),
            Ok(Poll::Pending) => {
                if self.eos {
                    Ok(crate::ops::MediaPoll::EndOfStream)
                } else {
                    Ok(crate::ops::MediaPoll::Pending)
                }
            }
            Ok(Poll::EndOfStream) => Ok(crate::ops::MediaPoll::EndOfStream),
            Err(error) if is_again(&error) => Ok(crate::ops::MediaPoll::Pending),
            Err(error) if is_end_of_stream(&error) => Ok(crate::ops::MediaPoll::EndOfStream),
            Err(error) => Err(map_av_error(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{codec_from_name, create_decoder, create_encoder, HwPreference};
    use dg_media_avcodec::{CodecId, Image, ImageInfo};

    #[test]
    fn codec_names_include_video_codecs() {
        assert_eq!(codec_from_name(Some("jpeg")), Ok(CodecId::Jpeg));
        assert_eq!(codec_from_name(Some("mjpeg")), Ok(CodecId::Mjpeg));
        assert_eq!(codec_from_name(Some("h264")), Ok(CodecId::H264));
    }

    #[test]
    fn hardware_preference_accepts_aliases() {
        assert_eq!(
            HwPreference::parse(Some("rknpu")),
            Ok(HwPreference::Rockchip)
        );
        assert_eq!(HwPreference::parse(Some("cuda")), Ok(HwPreference::Nvidia));
        assert_eq!(HwPreference::parse(Some("vaapi")), Ok(HwPreference::Intel));
        assert_eq!(HwPreference::parse(Some("amf")), Ok(HwPreference::Amd));
        assert_eq!(
            HwPreference::parse(Some("software")),
            Ok(HwPreference::Software)
        );
        assert!(HwPreference::parse(Some("mystery")).is_err());
    }

    #[test]
    fn jpeg_selection_uses_default_registry() {
        let image = Image::new_host_packed(ImageInfo::Rgb24, 2, 2, 0, 6, vec![0; 12], 1)
            .expect("valid JPEG image");
        assert!(create_encoder(CodecId::Jpeg, HwPreference::Auto, &image).is_ok());
    }

    #[test]
    fn h264_selection_reports_all_attempts_without_video_backends() {
        let error = match create_decoder(CodecId::H264, HwPreference::Auto) {
            Ok(_) => panic!("H264 must require an explicitly enabled backend"),
            Err(error) => error,
        };
        let message = error.to_string();
        assert!(message.contains("codec H264"), "{message}");
        assert!(message.contains("hardware preference Auto"), "{message}");
        assert!(message.contains("rkmpp"), "{message}");
        assert!(message.contains("ffmpeg"), "{message}");
        assert!(message.contains("codec-openh264"), "{message}");
    }

    #[test]
    fn h264_encoder_selection_reports_all_attempts_without_video_backends() {
        let image = Image::new_host_packed(ImageInfo::Rgb24, 2, 2, 0, 6, vec![0; 12], 1)
            .expect("valid H264 image");
        let error = match create_encoder(CodecId::H264, HwPreference::Auto, &image) {
            Ok(_) => panic!("H264 must require an explicitly enabled backend"),
            Err(error) => error,
        };
        let message = error.to_string();
        assert!(message.contains("codec H264"), "{message}");
        assert!(message.contains("hardware preference Auto"), "{message}");
        assert!(message.contains("x264"), "{message}");
        assert!(message.contains("codec-openh264"), "{message}");
    }
}
