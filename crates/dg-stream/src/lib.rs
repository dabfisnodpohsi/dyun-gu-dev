#![forbid(unsafe_code)]

//! Framework-native stream traits, metadata, and frame-sharing adapters.
//!
//! `dg-stream` bridges media frames into a stream/publisher/subscriber model
//! without hard-wiring a runtime or networking stack.

mod bridge;
mod error;
mod ids;
mod mock;
mod stream;
mod track;

pub use error::{Error, Result};
pub use ids::{PublishLease, StreamId, StreamKey, SubscriberId};
pub use mock::{MockPublisherSink, MockStreamManager, MockSubscriberSource};
pub use stream::{
    BackpressurePolicy, BootstrapMode, BootstrapPolicy, CoreAdaptersApi, DispatchResult,
    PublisherApi, PublisherOptions, PublisherSink, StreamManagerApi, StreamSnapshot, SubscriberApi,
    SubscriberOptions, SubscriberSource, SubscriberSourceSyncExt,
};
pub use track::{
    AacRtpPacketization, CodecConfigError, CodecConfigPayload, CodecConfigRequirement,
    CodecConfigView, CodecExtradata, CodecId, MediaKind, Rational32, TrackInfo, TrackReadiness,
};

pub use dg_media::MediaFrame;

#[cfg(feature = "cheetah")]
pub use bridge::{
    cheetah_avframe_to_media_frame, cheetah_track_info_to_media_frame,
    media_frame_to_cheetah_avframe, media_frame_to_cheetah_track_info, media_frame_to_frame,
};

#[cfg(not(feature = "cheetah"))]
pub use bridge::media_frame_to_frame;
