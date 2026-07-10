use std::collections::VecDeque;

use crate::MediaFrame;

/// Simple in-memory media source for tests.
#[derive(Debug, Default)]
pub struct MockMediaSource {
    frames: VecDeque<MediaFrame>,
}

impl MockMediaSource {
    pub fn new(frames: impl IntoIterator<Item = MediaFrame>) -> Self {
        Self {
            frames: frames.into_iter().collect(),
        }
    }

    pub fn push(&mut self, frame: MediaFrame) {
        self.frames.push_back(frame);
    }

    pub fn poll(&mut self) -> Option<MediaFrame> {
        self.frames.pop_front()
    }
}

/// Simple in-memory media sink for tests.
#[derive(Debug, Default)]
pub struct MockMediaSink {
    frames: Vec<MediaFrame>,
}

impl MockMediaSink {
    pub fn push(&mut self, frame: MediaFrame) {
        self.frames.push(frame);
    }

    pub fn frames(&self) -> &[MediaFrame] {
        &self.frames
    }

    pub fn into_frames(self) -> Vec<MediaFrame> {
        self.frames
    }
}
