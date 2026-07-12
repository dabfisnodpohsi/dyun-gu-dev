#![forbid(unsafe_code)]

//! Framework-native stream traits, metadata, and frame-sharing adapters.
//!
//! `dg-stream` bridges media frames into a stream/publisher/subscriber model
//! without hard-wiring a runtime or networking stack.

mod bridge;
mod connector;
mod elements;
#[cfg(feature = "cheetah")]
mod embedded;
mod error;
mod hub;
mod ids;
mod mock;
mod stream;
mod track;

pub use connector::{open_pull, open_push, PullEndpoint, StreamProtocol};
pub use error::{Error, Result};
pub use hub::{HubPublisher, HubSubscriber, MemoryStreamHub, KEYFRAME_TAG, MEDIA_TAG};
pub use ids::{PublishLease, StreamId, StreamKey, SubscriberId};
pub use mock::{MockPublisherSink, MockStreamManager, MockSubscriberSource};
pub use stream::{
    BackpressurePolicy, BootstrapMode, BootstrapPolicy, CoreAdaptersApi, DispatchResult,
    MediaFilter, PublisherApi, PublisherOptions, PublisherSink, StreamManagerApi, StreamSnapshot,
    SubscriberApi, SubscriberOptions, SubscriberSource, SubscriberSourceSyncExt,
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
    CheetahPublisherSinkAdapter, CheetahSubscriberSourceAdapter,
};

#[cfg(feature = "cheetah")]
pub use connector::{install_cheetah_connector, CheetahRuntimeConnector};
#[cfg(feature = "cheetah")]
pub use embedded::{install_embedded_cheetah_connector, EmbeddedCheetahRuntimeConnector};

#[cfg(not(feature = "cheetah"))]
pub use bridge::media_frame_to_frame;
