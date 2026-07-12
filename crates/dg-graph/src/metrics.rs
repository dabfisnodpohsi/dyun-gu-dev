use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

#[derive(Debug, Default)]
pub(crate) struct ElementMetrics {
    packets_processed: AtomicU64,
    packets_received: AtomicU64,
    packets_sent: AtomicU64,
    processing_latency_ns: AtomicU64,
    processing_latency_max_ns: AtomicU64,
    queue_depth: AtomicUsize,
    max_queue_depth: AtomicUsize,
    drop_count: AtomicU64,
    backpressure_count: AtomicU64,
}

impl ElementMetrics {
    pub(crate) fn record_received(&self) {
        self.packets_received.fetch_add(1, Ordering::Relaxed);
        self.packets_processed.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_source_packet(&self) {
        self.packets_processed.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_sent(&self) {
        self.packets_sent.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_latency(&self, duration: Duration) {
        let nanos = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        self.processing_latency_ns
            .fetch_add(nanos, Ordering::Relaxed);
        self.processing_latency_max_ns
            .fetch_max(nanos, Ordering::Relaxed);
    }

    pub(crate) fn record_queue_depth(&self, depth: usize) {
        self.queue_depth.store(depth, Ordering::Relaxed);
        self.max_queue_depth.fetch_max(depth, Ordering::Relaxed);
    }

    pub(crate) fn record_drop(&self) {
        self.drop_count.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_backpressure(&self) {
        self.backpressure_count.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self) -> ElementMetricsSnapshot {
        let packets_processed = self.packets_processed.load(Ordering::Relaxed);
        let processing_latency_ns = self.processing_latency_ns.load(Ordering::Relaxed);
        ElementMetricsSnapshot {
            packets_processed,
            packets_received: self.packets_received.load(Ordering::Relaxed),
            packets_sent: self.packets_sent.load(Ordering::Relaxed),
            processing_latency_ns,
            processing_latency_avg_ns: processing_latency_ns
                .checked_div(packets_processed)
                .unwrap_or_default(),
            processing_latency_max_ns: self.processing_latency_max_ns.load(Ordering::Relaxed),
            queue_depth: self.queue_depth.load(Ordering::Relaxed),
            max_queue_depth: self.max_queue_depth.load(Ordering::Relaxed),
            drop_count: self.drop_count.load(Ordering::Relaxed),
            backpressure_count: self.backpressure_count.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ElementMetricsSnapshot {
    pub packets_processed: u64,
    pub packets_received: u64,
    pub packets_sent: u64,
    pub processing_latency_ns: u64,
    pub processing_latency_avg_ns: u64,
    pub processing_latency_max_ns: u64,
    pub queue_depth: usize,
    pub max_queue_depth: usize,
    pub drop_count: u64,
    pub backpressure_count: u64,
}

/// Receives per-node snapshots for future exporters such as Prometheus.
pub trait MetricsSink: Send + Sync {
    fn record(&self, node: &str, metrics: &ElementMetricsSnapshot);
}
