use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::error::{Error, Result};
use crate::ids::{StreamId, StreamKey, SubscriberId};
use crate::stream::{
    DispatchResult, PublisherOptions, PublisherSink, StreamManagerApi, StreamSnapshot,
    SubscriberOptions, SubscriberSource,
};
use crate::track::TrackInfo;
use dg_media::MediaFrame;

/// In-memory publisher sink for tests.
#[derive(Debug, Default, Clone)]
pub struct MockPublisherSink {
    inner: Arc<Mutex<MockPublisherState>>,
}

#[derive(Debug, Default)]
struct MockPublisherState {
    tracks: Vec<TrackInfo>,
    frames: Vec<Arc<MediaFrame>>,
    keyframe_requests: u64,
    closed: bool,
}

impl MockPublisherSink {
    pub fn frames(&self) -> Vec<Arc<MediaFrame>> {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.frames.clone()
    }
}

impl PublisherSink for MockPublisherSink {
    fn update_tracks(&self, tracks: Vec<TrackInfo>) -> Result<()> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.tracks = tracks;
        Ok(())
    }

    fn push_frame(&self, frame: Arc<MediaFrame>) -> Result<DispatchResult> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.closed {
            return Ok(DispatchResult::RejectedClosed);
        }
        guard.frames.push(frame);
        Ok(DispatchResult::Accepted)
    }

    fn close(&self) -> Result<()> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.closed = true;
        Ok(())
    }

    fn take_keyframe_requests(&self) -> u64 {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let value = guard.keyframe_requests;
        guard.keyframe_requests = 0;
        value
    }
}

/// In-memory subscriber source for tests.
#[derive(Debug, Default)]
pub struct MockSubscriberSource {
    id: SubscriberId,
    frames: VecDeque<Arc<MediaFrame>>,
    closed: bool,
}

impl MockSubscriberSource {
    pub fn new(id: u64, frames: impl IntoIterator<Item = Arc<MediaFrame>>) -> Self {
        Self {
            id: SubscriberId(id),
            frames: frames.into_iter().collect(),
            closed: false,
        }
    }
}

#[async_trait]
impl SubscriberSource for MockSubscriberSource {
    async fn recv(&mut self) -> Result<Option<Arc<MediaFrame>>> {
        if self.closed {
            return Ok(None);
        }
        Ok(self.frames.pop_front())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed = true;
        Ok(())
    }

    fn id(&self) -> SubscriberId {
        self.id
    }
}

#[derive(Debug, Default)]
struct StreamState {
    tracks: Vec<TrackInfo>,
    publisher_active: bool,
    subscribers: usize,
}

/// Simple in-memory stream manager for tests.
#[derive(Debug, Default, Clone)]
pub struct MockStreamManager {
    streams: Arc<Mutex<HashMap<StreamKey, StreamState>>>,
}

impl MockStreamManager {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl StreamManagerApi for MockStreamManager {
    async fn open_publisher(
        &self,
        stream_key: StreamKey,
        _options: PublisherOptions,
    ) -> Result<Box<dyn PublisherSink>> {
        let mut guard = self
            .streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let state = guard.entry(stream_key).or_default();
        state.publisher_active = true;
        Ok(Box::new(MockPublisherSink::default()))
    }

    async fn open_subscriber(
        &self,
        stream_key: StreamKey,
        _options: SubscriberOptions,
    ) -> Result<Box<dyn SubscriberSource>> {
        let mut guard = self
            .streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let state = guard.entry(stream_key).or_default();
        state.subscribers = state.subscribers.saturating_add(1);
        Ok(Box::new(MockSubscriberSource::default()))
    }

    async fn list_streams(&self) -> Result<Vec<StreamSnapshot>> {
        let guard = self
            .streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(guard
            .iter()
            .enumerate()
            .map(|(index, (key, state))| StreamSnapshot {
                stream_id: StreamId(index as u64),
                key: key.clone(),
                publisher_active: state.publisher_active,
                subscriber_count: state.subscribers,
                tracks: state.tracks.clone(),
            })
            .collect())
    }

    async fn get_stream(&self, stream_key: &StreamKey) -> Result<Option<StreamSnapshot>> {
        let guard = self
            .streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(guard.get(stream_key).map(|state| StreamSnapshot {
            stream_id: StreamId(0),
            key: stream_key.clone(),
            publisher_active: state.publisher_active,
            subscriber_count: state.subscribers,
            tracks: state.tracks.clone(),
        }))
    }

    async fn request_keyframe(&self, stream_key: &StreamKey) -> Result<()> {
        let guard = self
            .streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.contains_key(stream_key) {
            Ok(())
        } else {
            Err(Error::InvalidArgument("unknown stream key".to_string()))
        }
    }

    async fn close_idle_publishers(&self, _max_idle_secs: u64) -> Result<usize> {
        Ok(0)
    }
}
