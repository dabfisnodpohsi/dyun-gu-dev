use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use tracing::debug;

use crate::error::{Error, Result};
use crate::ids::SubscriberId;
use crate::stream::{
    BackpressurePolicy, DispatchResult, MediaFilter, PublisherOptions, PublisherSink,
    SubscriberOptions, SubscriberSource,
};
use crate::track::{TrackInfo, TrackReadiness};
use dg_media::MediaFrame;

const RECV_POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Tag key marking a frame as a random access point.
pub const KEYFRAME_TAG: &str = "keyframe";
/// Tag key carrying the media kind (`video` / `audio`) of a frame.
pub const MEDIA_TAG: &str = "media";

fn is_keyframe(frame: &MediaFrame) -> bool {
    frame
        .meta
        .tags
        .get(KEYFRAME_TAG)
        .is_some_and(|value| value == "true")
}

fn passes_filter(frame: &MediaFrame, filter: MediaFilter) -> bool {
    match frame.meta.tags.get(MEDIA_TAG).map(String::as_str) {
        Some("video") => filter.enable_video,
        Some("audio") => filter.enable_audio,
        _ => true,
    }
}

#[derive(Debug)]
struct SubscriberQueue {
    queue: VecDeque<Arc<MediaFrame>>,
    capacity: usize,
    policy: BackpressurePolicy,
    filter: MediaFilter,
    overflowed: bool,
    dropping_until_keyframe: bool,
}

#[derive(Debug, Default)]
struct StreamState {
    tracks: Vec<TrackInfo>,
    subscribers: HashMap<SubscriberId, SubscriberQueue>,
    publisher_closed: bool,
    keyframe_requests: u64,
}

#[derive(Debug, Default)]
struct StreamCore {
    state: Mutex<StreamState>,
    frame_ready: Condvar,
}

impl StreamCore {
    fn lock(&self) -> MutexGuard<'_, StreamState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// In-process stream hub backing the `mock://` scheme of the stream elements.
///
/// The hub is a Sans-I/O test double for a real media server: publishers and
/// subscribers exchange frames through bounded per-subscriber queues with the
/// configured [`BackpressurePolicy`], and publisher close is propagated to all
/// subscribers as a clean end of stream.
#[derive(Debug, Default)]
pub struct MemoryStreamHub {
    streams: Mutex<HashMap<String, Arc<StreamCore>>>,
    next_subscriber: AtomicU64,
}

static GLOBAL_HUB: OnceLock<MemoryStreamHub> = OnceLock::new();

impl MemoryStreamHub {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process-wide hub used by graph elements with `mock://` URLs.
    pub fn global() -> &'static Self {
        GLOBAL_HUB.get_or_init(Self::new)
    }

    fn stream(&self, url: &str) -> Arc<StreamCore> {
        let mut guard = self
            .streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Arc::clone(guard.entry(url.to_string()).or_default())
    }

    /// Opens a publisher for the stream at `url`.
    pub fn publish(&self, url: &str, _options: PublisherOptions) -> Result<HubPublisher> {
        let core = self.stream(url);
        {
            let mut state = core.lock();
            state.publisher_closed = false;
        }
        Ok(HubPublisher { core })
    }

    /// Opens a subscriber for the stream at `url`.
    pub fn subscribe(&self, url: &str, options: SubscriberOptions) -> Result<HubSubscriber> {
        if options.queue_capacity == 0 {
            return Err(Error::InvalidArgument(
                "subscriber queue_capacity must be greater than zero".to_string(),
            ));
        }
        let core = self.stream(url);
        let id = SubscriberId(self.next_subscriber.fetch_add(1, Ordering::Relaxed));
        {
            let mut state = core.lock();
            state.subscribers.insert(
                id,
                SubscriberQueue {
                    queue: VecDeque::new(),
                    capacity: options.queue_capacity,
                    policy: options.backpressure,
                    filter: options.media_filter,
                    overflowed: false,
                    dropping_until_keyframe: false,
                },
            );
        }
        Ok(HubSubscriber {
            core,
            id,
            closed: false,
        })
    }

    /// Current track metadata announced on the stream at `url`.
    pub fn tracks(&self, url: &str) -> Vec<TrackInfo> {
        self.stream(url).lock().tracks.clone()
    }

    /// Number of active subscribers on the stream at `url`.
    pub fn subscriber_count(&self, url: &str) -> usize {
        self.stream(url).lock().subscribers.len()
    }

    /// Requests a keyframe from the publisher of the stream at `url`.
    pub fn request_keyframe(&self, url: &str) -> Result<()> {
        let core = self.stream(url);
        let mut state = core.lock();
        state.keyframe_requests = state.keyframe_requests.saturating_add(1);
        Ok(())
    }
}

/// Publisher endpoint of the in-memory hub.
pub struct HubPublisher {
    core: Arc<StreamCore>,
}

impl PublisherSink for HubPublisher {
    fn update_tracks(&self, tracks: Vec<TrackInfo>) -> Result<()> {
        for track in &tracks {
            if track.readiness == TrackReadiness::Ready {
                track
                    .validate_codec_config()
                    .map_err(|err| Error::Media(err.to_string()))?;
            }
        }
        let mut state = self.core.lock();
        if state.publisher_closed {
            return Err(Error::Closed);
        }
        state.tracks = tracks;
        Ok(())
    }

    fn push_frame(&self, frame: Arc<MediaFrame>) -> Result<DispatchResult> {
        let mut state = self.core.lock();
        if state.publisher_closed {
            return Ok(DispatchResult::RejectedClosed);
        }
        let mut enqueued = false;
        let mut dropped = false;
        for subscriber in state.subscribers.values_mut() {
            if subscriber.overflowed || !passes_filter(&frame, subscriber.filter) {
                continue;
            }
            if subscriber.dropping_until_keyframe {
                if is_keyframe(&frame) {
                    subscriber.dropping_until_keyframe = false;
                } else {
                    dropped = true;
                    continue;
                }
            }
            if subscriber.queue.len() >= subscriber.capacity {
                match subscriber.policy {
                    BackpressurePolicy::DropDroppableFirst => {
                        let position = subscriber
                            .queue
                            .iter()
                            .position(|queued| !is_keyframe(queued));
                        match position {
                            Some(index) => {
                                subscriber.queue.remove(index);
                            }
                            None => {
                                subscriber.queue.pop_front();
                            }
                        }
                        dropped = true;
                        subscriber.queue.push_back(Arc::clone(&frame));
                        enqueued = true;
                    }
                    BackpressurePolicy::DropUntilNextKeyframe => {
                        dropped = true;
                        if is_keyframe(&frame) {
                            subscriber.queue.clear();
                            subscriber.queue.push_back(Arc::clone(&frame));
                            enqueued = true;
                        } else {
                            subscriber.dropping_until_keyframe = true;
                        }
                    }
                    BackpressurePolicy::DisconnectOnOverflow => {
                        subscriber.overflowed = true;
                        subscriber.queue.clear();
                        dropped = true;
                    }
                }
            } else {
                subscriber.queue.push_back(Arc::clone(&frame));
                enqueued = true;
            }
        }
        drop(state);
        self.core.frame_ready.notify_all();
        if enqueued || !dropped {
            Ok(DispatchResult::Accepted)
        } else {
            debug!("hub publisher dropped frame on all subscribers by policy");
            Ok(DispatchResult::DroppedByPolicy)
        }
    }

    fn close(&self) -> Result<()> {
        let mut state = self.core.lock();
        state.publisher_closed = true;
        drop(state);
        self.core.frame_ready.notify_all();
        Ok(())
    }

    fn take_keyframe_requests(&self) -> u64 {
        let mut state = self.core.lock();
        let value = state.keyframe_requests;
        state.keyframe_requests = 0;
        value
    }
}

impl Drop for HubPublisher {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Subscriber endpoint of the in-memory hub.
pub struct HubSubscriber {
    core: Arc<StreamCore>,
    id: SubscriberId,
    closed: bool,
}

#[async_trait]
impl SubscriberSource for HubSubscriber {
    async fn recv(&mut self) -> Result<Option<Arc<MediaFrame>>> {
        if self.closed {
            return Ok(None);
        }
        let mut state = self.core.lock();
        loop {
            let Some(subscriber) = state.subscribers.get_mut(&self.id) else {
                return Ok(None);
            };
            if subscriber.overflowed {
                state.subscribers.remove(&self.id);
                self.closed = true;
                return Err(Error::Overflow(
                    "subscriber disconnected: queue overflow".to_string(),
                ));
            }
            if let Some(frame) = subscriber.queue.pop_front() {
                return Ok(Some(frame));
            }
            if state.publisher_closed {
                return Ok(None);
            }
            state = self
                .core
                .frame_ready
                .wait_timeout(state, RECV_POLL_INTERVAL)
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .0;
        }
    }

    async fn close(&mut self) -> Result<()> {
        if !self.closed {
            self.closed = true;
            let mut state = self.core.lock();
            state.subscribers.remove(&self.id);
        }
        Ok(())
    }

    fn id(&self) -> SubscriberId {
        self.id
    }
}
