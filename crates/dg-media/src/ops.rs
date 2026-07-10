//! Sans-I/O cores for the media elements.
//!
//! Each core is a pure submit/poll state machine over [`MediaFrame`]s: no
//! blocking I/O, no threads, no channels. Drivers (graph elements, tests)
//! feed frames in via `submit_*` and drain results via `poll`.

use std::collections::VecDeque;

use dg_core::{DataFormat, DataType, Error, Result};

use crate::{MediaFrame, MediaFrameKind};

/// Non-blocking poll result shared by all media cores.
#[derive(Debug)]
pub enum MediaPoll {
    Ready(MediaFrame),
    Pending,
    EndOfStream,
}

fn image_geometry(frame: &MediaFrame, element: &str) -> Result<(usize, usize, usize)> {
    match frame.shape.as_slice() {
        [height, width, channels] => Ok((*height, *width, *channels)),
        other => Err(Error::Media(format!(
            "{element}: expected [height, width, channels] image shape, got {other:?}"
        ))),
    }
}

fn checked_pixel_count(
    height: usize,
    width: usize,
    channels: usize,
    element: &str,
) -> Result<usize> {
    height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| Error::Media(format!("{element}: image dimensions overflow")))
}

/// Sans-I/O decode core.
///
/// Without a hardware codec attached, packets are treated as recorded raw
/// frames: each submitted bitstream payload must contain exactly
/// `width * height * channels` bytes and is re-labelled as an image frame,
/// sharing the underlying buffer (no byte copy).
#[derive(Debug)]
pub struct DecodeCore {
    width: usize,
    height: usize,
    channels: usize,
    ready: VecDeque<MediaFrame>,
    eos: bool,
}

impl DecodeCore {
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Self {
            width,
            height,
            channels,
            ready: VecDeque::new(),
            eos: false,
        }
    }

    pub fn submit_packet(&mut self, frame: MediaFrame) -> Result<()> {
        if self.eos {
            return Err(Error::Media(
                "media_decode: packet submitted after end of stream".to_string(),
            ));
        }
        if frame.is_end_of_stream() {
            self.eos = true;
            return Ok(());
        }
        let expected = checked_pixel_count(self.height, self.width, self.channels, "media_decode")?;
        if frame.buffer.len() != expected {
            return Err(Error::Media(format!(
                "media_decode: recorded frame has {} bytes, expected {} ({}x{}x{})",
                frame.buffer.len(),
                expected,
                self.height,
                self.width,
                self.channels
            )));
        }
        let mut image = MediaFrame::new(
            MediaFrameKind::Image,
            DataType::U8,
            DataFormat::NHWC,
            vec![self.height, self.width, self.channels],
            frame.device,
            frame.domain,
            frame.buffer,
        );
        image.meta = frame.meta;
        self.ready.push_back(image);
        Ok(())
    }

    pub fn submit_end_of_stream(&mut self) {
        self.eos = true;
    }

    pub fn poll(&mut self) -> MediaPoll {
        match self.ready.pop_front() {
            Some(frame) => MediaPoll::Ready(frame),
            None if self.eos => MediaPoll::EndOfStream,
            None => MediaPoll::Pending,
        }
    }
}

/// Sans-I/O encode core.
///
/// The inverse of [`DecodeCore`]: image frames are re-labelled as flat
/// bitstream payloads, sharing the underlying buffer (no byte copy).
#[derive(Debug, Default)]
pub struct EncodeCore {
    ready: VecDeque<MediaFrame>,
    eos: bool,
}

impl EncodeCore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn submit_image(&mut self, frame: MediaFrame) -> Result<()> {
        if self.eos {
            return Err(Error::Media(
                "media_encode: frame submitted after end of stream".to_string(),
            ));
        }
        if frame.is_end_of_stream() {
            self.eos = true;
            return Ok(());
        }
        let len = frame.buffer.len();
        let mut packet = MediaFrame::new(
            MediaFrameKind::Tensor,
            DataType::U8,
            DataFormat::N,
            vec![len],
            frame.device,
            frame.domain,
            frame.buffer,
        );
        packet.meta = frame.meta;
        self.ready.push_back(packet);
        Ok(())
    }

    pub fn submit_end_of_stream(&mut self) {
        self.eos = true;
    }

    pub fn poll(&mut self) -> MediaPoll {
        match self.ready.pop_front() {
            Some(frame) => MediaPoll::Ready(frame),
            None if self.eos => MediaPoll::EndOfStream,
            None => MediaPoll::Pending,
        }
    }
}

/// Sans-I/O nearest-neighbour resize core for interleaved `u8` images.
#[derive(Debug)]
pub struct ResizeCore {
    dst_width: usize,
    dst_height: usize,
    ready: VecDeque<MediaFrame>,
    eos: bool,
}

impl ResizeCore {
    pub fn new(dst_width: usize, dst_height: usize) -> Self {
        Self {
            dst_width,
            dst_height,
            ready: VecDeque::new(),
            eos: false,
        }
    }

    pub fn submit_image(&mut self, frame: MediaFrame) -> Result<()> {
        if self.eos {
            return Err(Error::Media(
                "media_resize: frame submitted after end of stream".to_string(),
            ));
        }
        if frame.is_end_of_stream() {
            self.eos = true;
            return Ok(());
        }
        let (src_height, src_width, channels) = image_geometry(&frame, "media_resize")?;
        if src_height == 0 || src_width == 0 || self.dst_height == 0 || self.dst_width == 0 {
            return Err(Error::Media(
                "media_resize: image dimensions must be non-zero".to_string(),
            ));
        }
        let expected = checked_pixel_count(src_height, src_width, channels, "media_resize")?;
        let src = frame.buffer.read_bytes();
        if src.len() != expected {
            return Err(Error::Media(format!(
                "media_resize: buffer has {} bytes, expected {expected}",
                src.len()
            )));
        }
        let out_len =
            checked_pixel_count(self.dst_height, self.dst_width, channels, "media_resize")?;
        let mut dst = vec![0_u8; out_len];
        for dst_y in 0..self.dst_height {
            let src_y = dst_y * src_height / self.dst_height;
            for dst_x in 0..self.dst_width {
                let src_x = dst_x * src_width / self.dst_width;
                let src_offset = (src_y * src_width + src_x) * channels;
                let dst_offset = (dst_y * self.dst_width + dst_x) * channels;
                dst[dst_offset..dst_offset + channels]
                    .copy_from_slice(&src[src_offset..src_offset + channels]);
            }
        }
        let mut resized = MediaFrame::from_host_bytes(
            MediaFrameKind::Image,
            DataType::U8,
            DataFormat::NHWC,
            vec![self.dst_height, self.dst_width, channels],
            frame.device,
            dst,
        )?;
        resized.meta = frame.meta;
        self.ready.push_back(resized);
        Ok(())
    }

    pub fn submit_end_of_stream(&mut self) {
        self.eos = true;
    }

    pub fn poll(&mut self) -> MediaPoll {
        match self.ready.pop_front() {
            Some(frame) => MediaPoll::Ready(frame),
            None if self.eos => MediaPoll::EndOfStream,
            None => MediaPoll::Pending,
        }
    }
}

/// Rectangle overlay drawn by [`OsdCore`], in pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OsdBox {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

/// Sans-I/O on-screen-display core: draws rectangle borders on `u8` images.
#[derive(Debug)]
pub struct OsdCore {
    boxes: Vec<OsdBox>,
    color: Vec<u8>,
    thickness: usize,
    ready: VecDeque<MediaFrame>,
    eos: bool,
}

impl OsdCore {
    pub fn new(boxes: Vec<OsdBox>, color: Vec<u8>, thickness: usize) -> Self {
        Self {
            boxes,
            color,
            thickness: thickness.max(1),
            ready: VecDeque::new(),
            eos: false,
        }
    }

    pub fn submit_image(&mut self, frame: MediaFrame) -> Result<()> {
        if self.eos {
            return Err(Error::Media(
                "media_osd: frame submitted after end of stream".to_string(),
            ));
        }
        if frame.is_end_of_stream() {
            self.eos = true;
            return Ok(());
        }
        let (height, width, channels) = image_geometry(&frame, "media_osd")?;
        if self.color.len() != channels {
            return Err(Error::Media(format!(
                "media_osd: color has {} components, image has {channels} channels",
                self.color.len()
            )));
        }
        let expected = checked_pixel_count(height, width, channels, "media_osd")?;
        let mut bytes = frame.buffer.read_bytes();
        if bytes.len() != expected {
            return Err(Error::Media(format!(
                "media_osd: buffer has {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        for osd_box in &self.boxes {
            self.draw_box(&mut bytes, width, height, channels, *osd_box);
        }
        let mut drawn = MediaFrame::from_host_bytes(
            MediaFrameKind::Image,
            DataType::U8,
            DataFormat::NHWC,
            vec![height, width, channels],
            frame.device,
            bytes,
        )?;
        drawn.meta = frame.meta;
        self.ready.push_back(drawn);
        Ok(())
    }

    fn draw_box(
        &self,
        bytes: &mut [u8],
        width: usize,
        height: usize,
        channels: usize,
        osd_box: OsdBox,
    ) {
        let x_end = (osd_box.x + osd_box.width).min(width);
        let y_end = (osd_box.y + osd_box.height).min(height);
        if osd_box.x >= x_end || osd_box.y >= y_end {
            return;
        }
        for y in osd_box.y..y_end {
            for x in osd_box.x..x_end {
                let on_border = y < osd_box.y + self.thickness
                    || y + self.thickness >= y_end
                    || x < osd_box.x + self.thickness
                    || x + self.thickness >= x_end;
                if on_border {
                    let offset = (y * width + x) * channels;
                    bytes[offset..offset + channels].copy_from_slice(&self.color);
                }
            }
        }
    }

    pub fn submit_end_of_stream(&mut self) {
        self.eos = true;
    }

    pub fn poll(&mut self) -> MediaPoll {
        match self.ready.pop_front() {
            Some(frame) => MediaPoll::Ready(frame),
            None if self.eos => MediaPoll::EndOfStream,
            None => MediaPoll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use dg_core::DeviceKind;

    use super::*;

    fn image_frame(height: usize, width: usize, channels: usize, fill: u8) -> MediaFrame {
        MediaFrame::from_host_bytes(
            MediaFrameKind::Image,
            DataType::U8,
            DataFormat::NHWC,
            vec![height, width, channels],
            DeviceKind::Cpu,
            vec![fill; height * width * channels],
        )
        .expect("build image frame")
    }

    #[test]
    fn decode_relabels_recorded_packets_without_copy() {
        let mut core = DecodeCore::new(2, 2, 1);
        let packet = MediaFrame::from_host_bytes(
            MediaFrameKind::Tensor,
            DataType::U8,
            DataFormat::N,
            vec![4],
            DeviceKind::Cpu,
            vec![9, 8, 7, 6],
        )
        .expect("build packet");
        let shared = packet.buffer.clone();
        core.submit_packet(packet).expect("submit packet");
        let MediaPoll::Ready(image) = core.poll() else {
            panic!("expected decoded image");
        };
        assert_eq!(image.kind, MediaFrameKind::Image);
        assert_eq!(image.shape, vec![2, 2, 1]);
        // Shared storage: decode must not copy the payload.
        assert!(shared.ref_count() >= 2);
        assert_eq!(image.buffer.read_bytes(), vec![9, 8, 7, 6]);
        core.submit_end_of_stream();
        assert!(matches!(core.poll(), MediaPoll::EndOfStream));
    }

    #[test]
    fn decode_rejects_wrong_payload_size() {
        let mut core = DecodeCore::new(2, 2, 3);
        let packet = MediaFrame::from_host_bytes(
            MediaFrameKind::Tensor,
            DataType::U8,
            DataFormat::N,
            vec![3],
            DeviceKind::Cpu,
            vec![0, 1, 2],
        )
        .expect("build packet");
        let err = core.submit_packet(packet).expect_err("expected size error");
        assert!(matches!(err, Error::Media(message) if message.contains("expected 12")));
    }

    #[test]
    fn encode_flattens_image_frames() {
        let mut core = EncodeCore::new();
        core.submit_image(image_frame(2, 3, 1, 5)).expect("submit");
        let MediaPoll::Ready(packet) = core.poll() else {
            panic!("expected encoded packet");
        };
        assert_eq!(packet.kind, MediaFrameKind::Tensor);
        assert_eq!(packet.shape, vec![6]);
        assert!(matches!(core.poll(), MediaPoll::Pending));
    }

    #[test]
    fn resize_nearest_neighbour_doubles_image() {
        let mut core = ResizeCore::new(4, 4);
        let frame = MediaFrame::from_host_bytes(
            MediaFrameKind::Image,
            DataType::U8,
            DataFormat::NHWC,
            vec![2, 2, 1],
            DeviceKind::Cpu,
            vec![1, 2, 3, 4],
        )
        .expect("build image");
        core.submit_image(frame).expect("submit image");
        let MediaPoll::Ready(resized) = core.poll() else {
            panic!("expected resized image");
        };
        assert_eq!(resized.shape, vec![4, 4, 1]);
        assert_eq!(
            resized.buffer.read_bytes(),
            vec![1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]
        );
    }

    #[test]
    fn osd_draws_border_and_leaves_interior() {
        let mut core = OsdCore::new(
            vec![OsdBox {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            }],
            vec![255],
            1,
        );
        core.submit_image(image_frame(4, 4, 1, 0)).expect("submit");
        let MediaPoll::Ready(drawn) = core.poll() else {
            panic!("expected osd output");
        };
        let bytes = drawn.buffer.read_bytes();
        assert_eq!(bytes[0], 255);
        assert_eq!(bytes[3], 255);
        assert_eq!(bytes[12], 255);
        // Interior pixel stays untouched.
        assert_eq!(bytes[5], 0);
    }

    #[test]
    fn osd_rejects_color_channel_mismatch() {
        let mut core = OsdCore::new(
            vec![OsdBox {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            }],
            vec![255, 0],
            1,
        );
        let err = core
            .submit_image(image_frame(2, 2, 3, 0))
            .expect_err("expected channel mismatch");
        assert!(matches!(err, Error::Media(message) if message.contains("channels")));
    }
}
