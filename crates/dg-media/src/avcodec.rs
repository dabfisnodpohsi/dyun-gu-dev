use dg_core::{Error, Result};

use crate::bridge::{
    avcodec_image_to_media_frame, avcodec_packet_to_media_frame, media_frame_to_avcodec_image,
    media_frame_to_avcodec_packet,
};
use crate::MediaFrame;

use dg_media_avcodec::{
    AvError, AvErrorContext, AvErrorKind, BitstreamFormat, CodecId, Decoder, DecoderConfig,
    Encoder, EncoderConfig, ImageOp, ImageProcessRequest, ImageProcessor, ImageProcessorConfig,
    MemoryDomain, Poll, Registry, RegistryBuilder, TimeBase, JPEG_BACKEND, ZUNE_BACKEND,
};

pub fn registry() -> Registry {
    RegistryBuilder::new()
        .with_backend(&JPEG_BACKEND)
        .with_backend(&ZUNE_BACKEND)
        .build()
}

pub fn codec_from_name(name: Option<&str>) -> Result<CodecId> {
    match name.unwrap_or("jpeg").to_ascii_lowercase().as_str() {
        "jpeg" => Ok(CodecId::Jpeg),
        "mjpeg" => Ok(CodecId::Mjpeg),
        other => Err(Error::Config(format!(
            "codec must be one of `jpeg` or `mjpeg`, got `{other}`"
        ))),
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

pub struct DecodeCore {
    decoder: Box<dyn Decoder>,
    codec: CodecId,
    eos: bool,
    pending_error: Option<Error>,
}

impl DecodeCore {
    pub fn new(codec: CodecId) -> Result<Self> {
        let config =
            DecoderConfig::new(codec, TimeBase::new(1, 25)).with_memory_domain(MemoryDomain::Host);
        let decoder = registry().create_decoder(&config).map_err(map_av_error)?;
        Ok(Self {
            decoder,
            codec,
            eos: false,
            pending_error: None,
        })
    }

    pub fn submit_packet(&mut self, frame: MediaFrame) -> Result<()> {
        if self.eos {
            return Err(Error::Media(
                "media_decode: packet submitted after end of stream".to_string(),
            ));
        }
        let packet =
            media_frame_to_avcodec_packet(frame, 0, self.codec, BitstreamFormat::JpegInterchange)?;
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
            Ok(Poll::Ready(image)) => Ok(crate::ops::MediaPoll::Ready(
                avcodec_image_to_media_frame(&image)?,
            )),
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
}

impl EncodeCore {
    pub fn new(codec: CodecId) -> Result<Self> {
        Ok(Self {
            encoder: None,
            codec,
            eos: false,
            pending_error: None,
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
            let config = EncoderConfig::new(
                self.codec,
                image.coded_width,
                image.coded_height,
                image.format,
                TimeBase::new(1, 25),
                1,
            )
            .with_memory_domain(MemoryDomain::Host);
            self.encoder = Some(registry().create_encoder(&config).map_err(map_av_error)?);
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
