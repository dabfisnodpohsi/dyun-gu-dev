#![forbid(unsafe_code)]

//! Framework-native media frames, zero-copy planning, and bridge utilities.
//!
//! `dg-media` owns the framework's media-side buffer envelope and the decision
//! logic for choosing zero-copy versus staging transfer paths.

#[cfg(feature = "avcodec")]
mod avcodec;
mod bridge;
mod elements;
mod frame;
mod mock;
mod ops;
mod planner;
mod stream_metadata;

#[cfg(feature = "avcodec")]
pub use avcodec::{DecodeCore as AvcodecDecodeCore, EncodeCore as AvcodecEncodeCore, HwPreference};
pub use frame::{MediaFrame, MediaFrameKind, MediaFrameMeta};
pub use mock::{MockMediaSink, MockMediaSource};
pub use ops::{DecodeCore, EncodeCore, MediaPoll, OsdBox, OsdCore, ResizeCore};
pub use planner::{
    preferred_memory_domain, CopyPath, FrameLayout, FrameTransferRequest, HandleKind, MemoryDtype,
    MemoryFormat, Subsampling, TransferMode, TransferReport, ZeroCopyPlan, ZeroCopyPlanner,
    ZeroCopyRequest,
};
pub use stream_metadata::{
    MediaStreamCodec, MediaStreamFormat, MediaStreamKind, MediaStreamMetadata, MediaStreamTimebase,
};

pub use dg_core::{
    DataFormat, DataType, DeployMode, DeviceKind, ExternalDropGuard, ExternalHandle, MemoryDomain,
    Result, Tensor, TensorDesc,
};

pub use bridge::{
    frame_to_tensor, graph_packet_to_media_frame, media_frame_to_graph_packet, tensor_to_frame,
    BridgedMediaFrame,
};

#[cfg(feature = "avcodec")]
pub use bridge::{
    avcodec_external_handle_to_core, avcodec_handle_to_buffer, avcodec_image_to_media_frame,
    avcodec_memory_domain_to_core, avcodec_packet_to_media_frame,
    avcodec_packet_to_media_frame_with_transfer, buffer_to_avcodec_handle,
    core_external_handle_to_avcodec, core_memory_domain_to_avcodec, import_avcodec_handle,
    media_frame_to_avcodec_image, media_frame_to_avcodec_image_with_transfer,
    media_frame_to_avcodec_packet, media_frame_to_avcodec_packet_with_transfer, ImportedBuffer,
};
