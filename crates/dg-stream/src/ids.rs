use serde::{Deserialize, Serialize};

/// Numeric stream identifier.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct StreamId(pub u64);

/// Structured stream key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StreamKey {
    pub namespace: String,
    pub path: String,
}

impl StreamKey {
    pub fn new(namespace: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            path: path.into(),
        }
    }
}

/// Numeric subscriber identifier.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct SubscriberId(pub u64);

/// Lease returned by publisher acquisition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublishLease {
    pub stream_id: StreamId,
    pub stream_key: StreamKey,
    pub lease_id: u64,
}
