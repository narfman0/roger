use std::sync::atomic::{AtomicU64, Ordering};

/// Process-lifetime counters for LLM responses. Cheap, lock-free; reset on restart.
#[derive(Default)]
pub struct Metrics {
    requests: AtomicU64,
    errors: AtomicU64,
    total_latency_ms: AtomicU64,
}

pub struct MetricsSnapshot {
    pub requests: u64,
    pub errors: u64,
    pub avg_latency_ms: u64,
}

impl Metrics {
    /// Record one completed response: its end-to-end latency and whether it succeeded.
    pub fn record(&self, latency_ms: u64, ok: bool) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        if !ok {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
        self.total_latency_ms.fetch_add(latency_ms, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let requests = self.requests.load(Ordering::Relaxed);
        let errors = self.errors.load(Ordering::Relaxed);
        let total = self.total_latency_ms.load(Ordering::Relaxed);
        MetricsSnapshot {
            requests,
            errors,
            avg_latency_ms: if requests > 0 { total / requests } else { 0 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_snapshot_is_zero() {
        let m = Metrics::default();
        let s = m.snapshot();
        assert_eq!((s.requests, s.errors, s.avg_latency_ms), (0, 0, 0));
    }

    #[test]
    fn records_counts_and_average_latency() {
        let m = Metrics::default();
        m.record(100, true);
        m.record(300, false);
        let s = m.snapshot();
        assert_eq!(s.requests, 2);
        assert_eq!(s.errors, 1);
        assert_eq!(s.avg_latency_ms, 200);
    }
}
