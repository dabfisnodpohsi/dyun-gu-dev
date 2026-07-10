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
    let tensor = match Arc::try_unwrap(payload) {
        Ok(PacketPayload::Tensor(tensor)) => Some(tensor),
        Ok(PacketPayload::EndOfStream) => None,
        Err(payload) => match payload.as_ref() {
            PacketPayload::Tensor(tensor) => Some(tensor.clone()),
            PacketPayload::EndOfStream => None,
        },
    };
    match tensor {
        Some(tensor) => {
            let mut frame = MediaFrame::from_tensor(tensor);
            frame.meta.pts = meta.sequence.try_into().ok();
            frame.meta.stream_id = meta.stream_id;
            frame.meta.tags = meta.tags;
            frame
        }
        None => MediaFrame::new(
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use dg_core::{CpuDevice, DataFormat, DataType, DeviceKind, Shape, Tensor, TensorDesc};
    use dg_graph::{Packet, PacketMeta};

    use super::graph_packet_to_media_frame;

    fn test_tensor() -> Tensor {
        let device = CpuDevice::new();
        let desc = TensorDesc::new(
            Shape::new([1, 4]),
            DataType::U8,
            DataFormat::NC,
            DeviceKind::Cpu,
        );
        let tensor = Tensor::allocate(&device, desc).expect("allocate test tensor");
        tensor
            .buffer()
            .write_from_slice(&[4, 3, 2, 1])
            .expect("write test tensor");
        tensor
    }

    #[test]
    fn graph_packet_bridge_preserves_shared_tensor_and_metadata() {
        let packet = Packet {
            meta: PacketMeta {
                sequence: 17,
                stream_id: Some("stream-a".to_string()),
                tags: BTreeMap::from([("kind".to_string(), "tensor".to_string())]),
            },
            payload: std::sync::Arc::new(dg_graph::PacketPayload::Tensor(test_tensor())),
        };
        let cloned_packet = packet.clone();

        let frame = graph_packet_to_media_frame(cloned_packet);

        assert!(!frame.is_end_of_stream());
        assert_eq!(frame.buffer.read_bytes(), vec![4, 3, 2, 1]);
        assert_eq!(frame.meta.pts, Some(17));
        assert_eq!(frame.meta.stream_id.as_deref(), Some("stream-a"));
        assert_eq!(
            frame.meta.tags.get("kind").map(String::as_str),
            Some("tensor")
        );
    }
}
