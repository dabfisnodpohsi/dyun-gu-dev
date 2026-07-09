use std::sync::Arc;

use dg_core::{Buffer, DataFormat, DataType, DeviceKind, MemoryDomain, Result, Tensor};
use dg_graph::{Packet, PacketPayload};

use crate::{MediaFrame, MediaFrameKind};

pub fn tensor_to_frame(tensor: Tensor) -> MediaFrame {
    MediaFrame::from_tensor(tensor)
}

pub fn frame_to_tensor(frame: MediaFrame) -> Result<Tensor> {
    frame.into_tensor()
}

pub fn graph_packet_to_media_frame(packet: Packet) -> MediaFrame {
    let Packet { meta, payload } = packet;
    match Arc::try_unwrap(payload) {
        Ok(PacketPayload::Tensor(tensor)) => {
            let mut frame = MediaFrame::from_tensor(tensor);
            frame.meta.pts = meta.sequence.try_into().ok();
            frame.meta.stream_id = meta.stream_id;
            frame.meta.tags = meta.tags;
            frame
        }
        Ok(PacketPayload::EndOfStream) | Err(_) => MediaFrame::new(
            MediaFrameKind::EndOfStream,
            DataType::U8,
            DataFormat::Auto,
            Vec::new(),
            DeviceKind::Cpu,
            MemoryDomain::Host,
            Buffer::allocate_host(DeviceKind::Cpu, 0),
        ),
    }
}

pub fn media_frame_to_graph_packet(frame: MediaFrame) -> Result<Packet> {
    if frame.is_end_of_stream() {
        return Ok(Packet::eos());
    }
    Ok(Packet::tensor(frame.into_tensor()?))
}

#[cfg(feature = "avcodec")]
pub fn avcodec_handle_to_buffer(
    handle: &dg_media_avcodec::BufferHandle,
    device: DeviceKind,
) -> Result<Buffer> {
    let staged = if handle.domain() == dg_media_avcodec::MemoryDomain::Host {
        handle.clone()
    } else {
        handle
            .stage_to_host(0)
            .map_err(|err| dg_core::Error::Buffer(format!("{err:?}")))?
    };
    let bytes = staged
        .host_bytes()
        .map_or_else(|| vec![0; staged.size()], |slice| slice.to_vec());
    Buffer::from_host_bytes(device, dg_core::BufferDesc::new(bytes.len(), 1), bytes)
}

#[cfg(feature = "avcodec")]
pub fn buffer_to_avcodec_handle(buffer: &Buffer) -> Result<dg_media_avcodec::BufferHandle> {
    Ok(dg_media_avcodec::BufferHandle::from_host_bytes(
        0,
        buffer.read_bytes(),
    ))
}

#[cfg(feature = "avcodec")]
pub fn avcodec_packet_to_media_frame(packet: &dg_media_avcodec::Packet) -> Result<MediaFrame> {
    let bytes = packet.host_bytes().map_or_else(
        |_| Vec::new(),
        |bytes| bytes.map_or_else(Vec::new, |slice| slice.to_vec()),
    );
    MediaFrame::from_host_bytes(
        MediaFrameKind::Tensor,
        DataType::U8,
        DataFormat::Auto,
        Vec::new(),
        DeviceKind::Cpu,
        bytes,
    )
}

#[cfg(feature = "avcodec")]
pub fn media_frame_to_avcodec_packet(
    frame: MediaFrame,
    stream_index: u32,
    codec: dg_media_avcodec::CodecId,
    bitstream_format: dg_media_avcodec::BitstreamFormat,
) -> Result<dg_media_avcodec::Packet> {
    let bytes = frame.buffer.into_host_bytes();
    Ok(dg_media_avcodec::Packet::from_host_bytes(
        u64::from(stream_index),
        codec,
        bitstream_format,
        bytes,
    ))
}

#[cfg(feature = "avcodec")]
pub fn avcodec_image_to_media_frame(image: &dg_media_avcodec::Image) -> Result<MediaFrame> {
    let host = if let Some(bytes) = image
        .plane_host_bytes(0)
        .map_err(|err| dg_core::Error::Buffer(format!("{err:?}")))?
    {
        bytes.to_vec()
    } else {
        let staged = image
            .memory
            .stage_to_host(0)
            .map_err(|err| dg_core::Error::Buffer(format!("{err:?}")))?;
        staged
            .host_bytes()
            .map_or_else(Vec::new, |slice| slice.to_vec())
    };
    MediaFrame::from_host_bytes(
        MediaFrameKind::Image,
        DataType::U8,
        DataFormat::Auto,
        vec![image.coded_height as usize, image.coded_width as usize],
        DeviceKind::Cpu,
        host,
    )
}

#[cfg(feature = "avcodec")]
pub fn media_frame_to_avcodec_image(
    frame: MediaFrame,
    stride_alignment: usize,
) -> Result<dg_media_avcodec::Image> {
    let height = frame.shape.first().copied().unwrap_or_default();
    let width = frame.shape.get(1).copied().unwrap_or_default();
    let coded_width = u32::try_from(width).unwrap_or(u32::MAX);
    let coded_height = u32::try_from(height).unwrap_or(u32::MAX);
    let format = match frame.shape.last().copied().unwrap_or_default() {
        3 => dg_media_avcodec::ImageInfo::Rgb24,
        4 => dg_media_avcodec::ImageInfo::Rgba,
        _ => dg_media_avcodec::ImageInfo::Gray8,
    };
    let bytes = frame.buffer.into_host_bytes();
    dg_media_avcodec::Image::new_host_packed(
        format,
        coded_width,
        coded_height,
        0,
        width,
        bytes,
        stride_alignment,
    )
    .map_err(|err| dg_core::Error::Buffer(format!("{err:?}")))
}

#[cfg(feature = "avcodec")]
#[allow(dead_code)]
pub fn avcodec_memory_domain_to_core(value: dg_media_avcodec::MemoryDomain) -> MemoryDomain {
    match value {
        dg_media_avcodec::MemoryDomain::Host => MemoryDomain::Host,
        dg_media_avcodec::MemoryDomain::DmaBuf => MemoryDomain::DmaBuf,
        dg_media_avcodec::MemoryDomain::DrmPrime => MemoryDomain::DrmPrime,
        dg_media_avcodec::MemoryDomain::VaapiSurface => MemoryDomain::VaapiSurface,
        dg_media_avcodec::MemoryDomain::CudaDevice => MemoryDomain::CudaDevice,
        dg_media_avcodec::MemoryDomain::MppBuffer => MemoryDomain::MppBuffer,
        dg_media_avcodec::MemoryDomain::OpaqueBackend => MemoryDomain::Opaque,
    }
}

#[cfg(feature = "avcodec")]
#[allow(dead_code)]
pub fn core_memory_domain_to_avcodec(value: MemoryDomain) -> dg_media_avcodec::MemoryDomain {
    match value {
        MemoryDomain::Host => dg_media_avcodec::MemoryDomain::Host,
        MemoryDomain::DmaBuf => dg_media_avcodec::MemoryDomain::DmaBuf,
        MemoryDomain::DrmPrime => dg_media_avcodec::MemoryDomain::DrmPrime,
        MemoryDomain::VaapiSurface => dg_media_avcodec::MemoryDomain::VaapiSurface,
        MemoryDomain::CudaDevice => dg_media_avcodec::MemoryDomain::CudaDevice,
        MemoryDomain::MppBuffer => dg_media_avcodec::MemoryDomain::MppBuffer,
        MemoryDomain::SophonDevice | MemoryDomain::Opaque => {
            dg_media_avcodec::MemoryDomain::OpaqueBackend
        }
    }
}

#[cfg(feature = "avcodec")]
#[allow(dead_code)]
pub fn avcodec_external_handle_to_core(
    value: dg_media_avcodec::ExternalHandle,
) -> crate::ExternalHandle {
    crate::ExternalHandle {
        fd: value.fd,
        raw: value.raw,
    }
}

#[cfg(feature = "avcodec")]
#[allow(dead_code)]
pub fn core_external_handle_to_avcodec(
    value: crate::ExternalHandle,
) -> dg_media_avcodec::ExternalHandle {
    dg_media_avcodec::ExternalHandle {
        fd: value.fd,
        raw: value.raw,
    }
}
