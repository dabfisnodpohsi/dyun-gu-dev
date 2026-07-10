use std::sync::Arc;

use dg_core::{Buffer, DataFormat, DataType, DeviceKind, MemoryDomain, Result, Tensor};
use dg_graph::{Packet, PacketPayload};

use crate::{MediaFrame, MediaFrameKind};

#[cfg(feature = "avcodec")]
use dg_core::{BufferDesc, ExternalDropGuard};
#[cfg(feature = "avcodec")]
use tracing::debug;

#[cfg(feature = "avcodec")]
use crate::CopyPath;

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
        Ok(
            PacketPayload::Detections(_)
            | PacketPayload::Classifications(_)
            | PacketPayload::Faces(_)
            | PacketPayload::Tracks(_)
            | PacketPayload::Ocr(_)
            | PacketPayload::EndOfStream,
        ) => None,
        Err(payload) => match payload.as_ref() {
            PacketPayload::Tensor(tensor) => Some(tensor.clone()),
            PacketPayload::Detections(_)
            | PacketPayload::Classifications(_)
            | PacketPayload::Faces(_)
            | PacketPayload::Tracks(_)
            | PacketPayload::Ocr(_)
            | PacketPayload::EndOfStream => None,
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

/// Result of importing an avcodec [`dg_media_avcodec::BufferHandle`] into a
/// dg-core [`Buffer`], with the actual transfer path taken.
#[cfg(feature = "avcodec")]
#[derive(Clone, Debug)]
pub struct ImportedBuffer {
    pub buffer: Buffer,
    pub zero_copy: bool,
    pub path: CopyPath,
}

/// Imports an avcodec buffer handle into a dg-core [`Buffer`] targeting
/// `target_domain`.
///
/// - Source domain equals `target_domain`: the handle is **shared** via
///   [`Buffer::from_external`] + [`ExternalDropGuard`] — the guard keeps a
///   clone of the avcodec handle alive until every clone of the returned
///   buffer is dropped (`copy_count == 0`). Host-readable bytes are exposed
///   through the buffer; non-host device memory is represented by the shared
///   external handle and is not host-readable through `read_bytes`.
/// - Otherwise an explicit **staging fallback** is taken through
///   `stage_to_host` / `stage_to` (`copy_count == 1` per domain crossing).
///   Missing staging support surfaces as [`dg_core::Error::Unsupported`]
///   rather than silently degrading.
///
/// The chosen path and copy count are logged and returned for diagnostics.
#[cfg(feature = "avcodec")]
pub fn import_avcodec_handle(
    handle: &dg_media_avcodec::BufferHandle,
    device: DeviceKind,
    target_domain: MemoryDomain,
) -> Result<ImportedBuffer> {
    let source_domain = avcodec_memory_domain_to_core(handle.domain());
    let (buffer, path) = if source_domain == target_domain {
        let buffer = share_avcodec_handle(handle, device, source_domain)?;
        (
            buffer,
            CopyPath {
                domains: vec![source_domain],
                copy_count: 0,
            },
        )
    } else {
        let staged = handle
            .stage_to(core_memory_domain_to_avcodec(target_domain), handle.id())
            .map_err(|err| match err {
                dg_media_avcodec::AvError::Unsupported => dg_core::Error::Unsupported(format!(
                    "no staging path from {source_domain:?} to {target_domain:?} for avcodec handle {}",
                    handle.id()
                )),
                other => dg_core::Error::Buffer(format!(
                    "staging avcodec handle {} from {source_domain:?} to {target_domain:?} failed: {other:?}",
                    handle.id()
                )),
            })?;
        let buffer = share_avcodec_handle(&staged, device, target_domain)?;
        (
            buffer,
            CopyPath {
                domains: vec![source_domain, target_domain],
                copy_count: 1,
            },
        )
    };
    let zero_copy = path.copy_count == 0;
    debug!(
        handle_id = handle.id(),
        source_domain = ?source_domain,
        target_domain = ?target_domain,
        copy_count = path.copy_count,
        zero_copy,
        path = ?path.domains,
        "imported avcodec buffer handle"
    );
    Ok(ImportedBuffer {
        buffer,
        zero_copy,
        path,
    })
}

/// Wraps an avcodec handle as a dg-core [`Buffer`] in the same memory domain.
///
/// The returned buffer holds an [`ExternalDropGuard`] owning a clone of the
/// avcodec handle, so the underlying decoder/encoder memory outlives every
/// clone of the buffer. Host-readable handles expose their bytes; device
/// handles only carry the shared [`dg_core::ExternalHandle`] token.
#[cfg(feature = "avcodec")]
fn share_avcodec_handle(
    handle: &dg_media_avcodec::BufferHandle,
    device: DeviceKind,
    domain: MemoryDomain,
) -> Result<Buffer> {
    let bytes = handle
        .host_bytes()
        .map_or_else(|| vec![0; handle.size()], <[u8]>::to_vec);
    let external = avcodec_external_handle_to_core(handle.external());
    let keepalive = handle.clone();
    let guard = ExternalDropGuard::new(move || drop(keepalive));
    Buffer::from_external(
        device,
        domain,
        BufferDesc::new(handle.size(), 1),
        external,
        bytes,
        guard,
    )
}

#[cfg(feature = "avcodec")]
pub fn avcodec_handle_to_buffer(
    handle: &dg_media_avcodec::BufferHandle,
    device: DeviceKind,
) -> Result<Buffer> {
    let source_domain = avcodec_memory_domain_to_core(handle.domain());
    let target_domain = if source_domain == MemoryDomain::Host {
        MemoryDomain::Host
    } else {
        source_domain
    };
    Ok(import_avcodec_handle(handle, device, target_domain)?.buffer)
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
pub fn avcodec_external_handle_to_core(
    value: dg_media_avcodec::ExternalHandle,
) -> crate::ExternalHandle {
    crate::ExternalHandle {
        fd: value.fd,
        raw: value.raw,
    }
}

#[cfg(feature = "avcodec")]
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

    #[cfg(feature = "avcodec")]
    mod avcodec {
        use dg_core::{DeviceKind, MemoryDomain};

        use crate::bridge::import_avcodec_handle;

        #[test]
        fn same_domain_device_handle_is_shared_without_copy() {
            let handle = dg_media_avcodec::BufferHandle::new(
                7,
                dg_media_avcodec::MemoryDomain::MppBuffer,
                16,
            );
            let imported =
                import_avcodec_handle(&handle, DeviceKind::RknnNpu, MemoryDomain::MppBuffer)
                    .expect("share device handle");
            assert!(imported.zero_copy);
            assert_eq!(imported.path.copy_count, 0);
            assert_eq!(imported.path.domains, vec![MemoryDomain::MppBuffer]);
            assert_eq!(imported.buffer.domain(), MemoryDomain::MppBuffer);
            assert_eq!(imported.buffer.len(), 16);

            // The imported buffer outlives the original avcodec handle.
            drop(handle);
            let clone = imported.buffer.clone();
            drop(imported);
            assert_eq!(clone.len(), 16);
        }

        #[test]
        fn host_handle_imports_host_bytes_without_staging() {
            let handle = dg_media_avcodec::BufferHandle::from_host_bytes(3, vec![1, 2, 3, 4]);
            let imported = import_avcodec_handle(&handle, DeviceKind::Cpu, MemoryDomain::Host)
                .expect("import host handle");
            assert!(imported.zero_copy);
            assert_eq!(imported.path.copy_count, 0);
            assert_eq!(imported.buffer.read_bytes(), vec![1, 2, 3, 4]);
        }

        #[test]
        fn missing_staging_path_fails_explicitly() {
            let handle = dg_media_avcodec::BufferHandle::new(
                9,
                dg_media_avcodec::MemoryDomain::MppBuffer,
                8,
            );
            let err = import_avcodec_handle(&handle, DeviceKind::Cpu, MemoryDomain::Host)
                .expect_err("expected unsupported staging path");
            assert!(matches!(err, dg_core::Error::Unsupported(message)
                if message.contains("MppBuffer") && message.contains("Host")));
        }

        #[test]
        fn staging_fallback_copies_through_registered_hook() {
            dg_media_avcodec::register_stage_to_host_hook(
                dg_media_avcodec::MemoryDomain::DmaBuf,
                |handle, dst| {
                    dst.fill(u8::try_from(handle.id()).unwrap_or(0));
                    Ok(())
                },
            );
            let handle =
                dg_media_avcodec::BufferHandle::new(5, dg_media_avcodec::MemoryDomain::DmaBuf, 4);
            let imported = import_avcodec_handle(&handle, DeviceKind::Cpu, MemoryDomain::Host)
                .expect("stage to host");
            assert!(!imported.zero_copy);
            assert_eq!(imported.path.copy_count, 1);
            assert_eq!(
                imported.path.domains,
                vec![MemoryDomain::DmaBuf, MemoryDomain::Host]
            );
            assert_eq!(imported.buffer.read_bytes(), vec![5, 5, 5, 5]);
        }
    }

    #[test]
    fn external_buffer_releases_ownership_once_after_last_clone() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let released = std::sync::Arc::new(AtomicUsize::new(0));
        let flag = released.clone();
        let guard = dg_core::ExternalDropGuard::new(move || {
            flag.fetch_add(1, Ordering::SeqCst);
        });
        let buffer = dg_core::Buffer::from_external(
            DeviceKind::Cpu,
            dg_core::MemoryDomain::DmaBuf,
            dg_core::BufferDesc::new(4, 1),
            dg_core::ExternalHandle::from_raw(42),
            vec![0; 4],
            guard,
        )
        .expect("import external buffer");

        let clone = buffer.clone();
        drop(buffer);
        assert_eq!(released.load(Ordering::SeqCst), 0);
        assert_eq!(clone.external().raw, 42);
        drop(clone);
        assert_eq!(released.load(Ordering::SeqCst), 1);
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
