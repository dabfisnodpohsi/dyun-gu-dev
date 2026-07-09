use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::ids::{PublishLease, StreamId, StreamKey, SubscriberId};
use crate::track::TrackInfo;
use dg_media::MediaFrame;

/// Backpressure policy for stream subscribers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackpressurePolicy {
    DropDroppableFirst,
    DropUntilNextKeyframe,
    DisconnectOnOverflow,
}

/// Bootstrap mode for late-joining subscribers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BootstrapMode {
    None,
    LiveTail,
    FullGop,
}

/// Bootstrap policy for late subscribers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapPolicy {
    pub mode: BootstrapMode,
    pub max_bootstrap_age_ms: Option<u64>,
    pub max_bootstrap_frames: usize,
    pub wait_for_next_random_access_point: bool,
}

impl BootstrapPolicy {
    pub const fn none() -> Self {
        Self {
            mode: BootstrapMode::None,
            max_bootstrap_age_ms: None,
            max_bootstrap_frames: 0,
            wait_for_next_random_access_point: false,
        }
    }

    pub const fn live_tail(max_bootstrap_frames: usize, max_bootstrap_age_ms: Option<u64>) -> Self {
        Self {
            mode: BootstrapMode::LiveTail,
            max_bootstrap_age_ms,
            max_bootstrap_frames,
            wait_for_next_random_access_point: true,
        }
    }

    pub const fn full_gop(max_bootstrap_frames: usize, max_bootstrap_age_ms: Option<u64>) -> Self {
        Self {
            mode: BootstrapMode::FullGop,
            max_bootstrap_age_ms,
            max_bootstrap_frames,
            wait_for_next_random_access_point: true,
        }
    }
}

impl Default for BootstrapPolicy {
    fn default() -> Self {
        Self {
            mode: BootstrapMode::LiveTail,
            max_bootstrap_age_ms: Some(1_500),
            max_bootstrap_frames: 150,
            wait_for_next_random_access_point: true,
        }
    }
}

/// Dispatch result from a publisher sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DispatchResult {
    Accepted,
    DroppedByPolicy,
    RejectedClosed,
}

/// Publisher options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublisherOptions {
    pub announce_tracks: bool,
}

impl Default for PublisherOptions {
    fn default() -> Self {
        Self {
            announce_tracks: true,
        }
    }
}

/// Subscriber options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubscriberOptions {
    pub queue_capacity: usize,
    pub backpressure: BackpressurePolicy,
    pub bootstrap_policy: BootstrapPolicy,
    pub media_filter: MediaFilter,
}

/// Media filter for subscriber selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaFilter {
    pub enable_video: bool,
    pub enable_audio: bool,
}

impl Default for MediaFilter {
    fn default() -> Self {
        Self {
            enable_video: true,
            enable_audio: true,
        }
    }
}

impl Default for SubscriberOptions {
    fn default() -> Self {
        Self {
            queue_capacity: 150,
            backpressure: BackpressurePolicy::DropDroppableFirst,
            bootstrap_policy: BootstrapPolicy::default(),
            media_filter: MediaFilter::default(),
        }
    }
}

/// Snapshot of a stream in the manager.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamSnapshot {
    pub stream_id: StreamId,
    pub key: StreamKey,
    pub publisher_active: bool,
    pub subscriber_count: usize,
    pub tracks: Vec<TrackInfo>,
}

/// Publisher sink boundary.
pub trait PublisherSink: Send + Sync {
    fn update_tracks(&self, tracks: Vec<TrackInfo>) -> Result<()>;
    fn push_frame(&self, frame: Arc<MediaFrame>) -> Result<DispatchResult>;
    fn close(&self) -> Result<()>;
    fn take_keyframe_requests(&self) -> u64;
}

/// Subscriber source boundary.
#[async_trait]
pub trait SubscriberSource: Send {
    async fn recv(&mut self) -> Result<Option<Arc<MediaFrame>>>;
    async fn close(&mut self) -> Result<()>;
    fn id(&self) -> SubscriberId;
}

/// Convenience extension for synchronous call sites.
pub trait SubscriberSourceSyncExt: SubscriberSource {
    fn recv_blocking(&mut self) -> Result<Option<Arc<MediaFrame>>> {
        futures::executor::block_on(self.recv())
    }

    fn close_blocking(&mut self) -> Result<()> {
        futures::executor::block_on(self.close())
    }
}

impl<T: SubscriberSource + ?Sized> SubscriberSourceSyncExt for T {}

/// Stream manager API boundary.
#[async_trait]
pub trait StreamManagerApi: Send + Sync {
    async fn open_publisher(
        &self,
        stream_key: StreamKey,
        options: PublisherOptions,
    ) -> Result<Box<dyn PublisherSink>>;

    async fn open_subscriber(
        &self,
        stream_key: StreamKey,
        options: SubscriberOptions,
    ) -> Result<Box<dyn SubscriberSource>>;

    async fn list_streams(&self) -> Result<Vec<StreamSnapshot>>;
    async fn get_stream(&self, stream_key: &StreamKey) -> Result<Option<StreamSnapshot>>;
    async fn request_keyframe(&self, stream_key: &StreamKey) -> Result<()>;
    async fn close_idle_publishers(&self, max_idle_secs: u64) -> Result<usize>;
}

/// Publisher API boundary.
#[async_trait]
pub trait PublisherApi: Send + Sync {
    async fn acquire_publisher(
        &self,
        stream_key: StreamKey,
        options: PublisherOptions,
    ) -> Result<(PublishLease, Box<dyn PublisherSink>)>;

    async fn release_publisher(&self, lease: &PublishLease) -> Result<()>;
}

/// Subscriber API boundary.
#[async_trait]
pub trait SubscriberApi: Send + Sync {
    async fn subscribe(
        &self,
        stream_key: StreamKey,
        options: SubscriberOptions,
    ) -> Result<Box<dyn SubscriberSource>>;
}

/// Core adapter API boundary.
#[async_trait]
pub trait CoreAdaptersApi: Send + Sync {
    async fn publish_frame(
        &self,
        stream_key: StreamKey,
        frame: Arc<MediaFrame>,
    ) -> Result<DispatchResult>;

    async fn update_tracks(&self, stream_key: StreamKey, tracks: Vec<TrackInfo>) -> Result<()>;

    async fn close_stream(&self, stream_key: &StreamKey) -> Result<()>;
}
