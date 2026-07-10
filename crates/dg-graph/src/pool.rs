//! Work-stealing thread pool backing graph execution.
//!
//! Each worker owns a queue; idle workers steal from the back of other
//! workers' queues (`try_push` / `try_pop` / `try_steal`).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::error::{Error, Result};

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Shared {
    queues: Vec<Mutex<VecDeque<Job>>>,
    pending: AtomicUsize,
    signal: Mutex<bool>,
    available: Condvar,
}

impl Shared {
    fn try_push(&self, job: Job) -> Result<()> {
        let mut target = 0;
        let mut shortest = usize::MAX;
        for (index, queue) in self.queues.iter().enumerate() {
            let guard = queue
                .lock()
                .map_err(|_| Error::Runtime("thread pool queue poisoned".to_string()))?;
            if guard.len() < shortest {
                shortest = guard.len();
                target = index;
            }
        }
        self.queues[target]
            .lock()
            .map_err(|_| Error::Runtime("thread pool queue poisoned".to_string()))?
            .push_back(job);
        self.pending.fetch_add(1, Ordering::SeqCst);
        let _guard = self
            .signal
            .lock()
            .map_err(|_| Error::Runtime("thread pool signal poisoned".to_string()))?;
        self.available.notify_one();
        Ok(())
    }

    fn try_pop(&self, worker: usize) -> Option<Job> {
        let job = self.queues[worker].lock().ok()?.pop_front();
        if job.is_some() {
            self.pending.fetch_sub(1, Ordering::SeqCst);
        }
        job
    }

    fn try_steal(&self, worker: usize) -> Option<Job> {
        for (index, queue) in self.queues.iter().enumerate() {
            if index == worker {
                continue;
            }
            if let Some(job) = queue.lock().ok().and_then(|mut guard| guard.pop_back()) {
                self.pending.fetch_sub(1, Ordering::SeqCst);
                return Some(job);
            }
        }
        None
    }
}

/// A fixed-size pool of worker threads with per-worker work-stealing queues.
pub struct ThreadPool {
    shared: Arc<Shared>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl ThreadPool {
    /// Creates a pool with `threads` workers. Errors when `threads` is zero.
    pub fn new(threads: usize) -> Result<Self> {
        if threads == 0 {
            return Err(Error::Config(
                "thread pool requires at least one worker".to_string(),
            ));
        }
        let shared = Arc::new(Shared {
            queues: (0..threads).map(|_| Mutex::new(VecDeque::new())).collect(),
            pending: AtomicUsize::new(0),
            signal: Mutex::new(false),
            available: Condvar::new(),
        });
        let workers = (0..threads)
            .map(|index| {
                let shared = shared.clone();
                thread::spawn(move || worker_loop(&shared, index))
            })
            .collect();
        Ok(Self { shared, workers })
    }

    pub fn threads(&self) -> usize {
        self.workers.len()
    }

    /// Schedules a job onto the least-loaded worker queue.
    pub fn spawn(&self, job: impl FnOnce() + Send + 'static) -> Result<()> {
        self.shared.try_push(Box::new(job))
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        if let Ok(mut shutdown) = self.shared.signal.lock() {
            *shutdown = true;
            self.shared.available.notify_all();
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker_loop(shared: &Shared, index: usize) {
    loop {
        if let Some(job) = shared.try_pop(index).or_else(|| shared.try_steal(index)) {
            job();
            continue;
        }
        let Ok(mut shutdown) = shared.signal.lock() else {
            return;
        };
        if *shutdown && shared.pending.load(Ordering::SeqCst) == 0 {
            return;
        }
        if shared.pending.load(Ordering::SeqCst) == 0 {
            let Ok(guard) = shared.available.wait(shutdown) else {
                return;
            };
            shutdown = guard;
            if *shutdown && shared.pending.load(Ordering::SeqCst) == 0 {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::Arc;
    use std::time::Duration;

    use super::ThreadPool;

    #[test]
    fn rejects_zero_workers() {
        assert!(ThreadPool::new(0).is_err());
    }

    #[test]
    fn runs_all_jobs_across_workers() {
        let pool = ThreadPool::new(4).expect("create pool");
        let counter = Arc::new(AtomicUsize::new(0));
        let (sender, receiver) = mpsc::channel();
        for _ in 0..64 {
            let counter = counter.clone();
            let sender = sender.clone();
            pool.spawn(move || {
                counter.fetch_add(1, Ordering::SeqCst);
                sender.send(()).expect("send completion");
            })
            .expect("spawn job");
        }
        drop(sender);
        for _ in 0..64 {
            receiver
                .recv_timeout(Duration::from_secs(5))
                .expect("job completes");
        }
        assert_eq!(counter.load(Ordering::SeqCst), 64);
    }

    #[test]
    fn idle_workers_steal_queued_jobs() {
        let pool = ThreadPool::new(4).expect("create pool");
        let (sender, receiver) = mpsc::channel();
        for _ in 0..32 {
            let sender = sender.clone();
            pool.spawn(move || {
                std::thread::sleep(Duration::from_millis(5));
                sender
                    .send(std::thread::current().id())
                    .expect("send thread id");
            })
            .expect("spawn job");
        }
        drop(sender);
        let mut threads = BTreeSet::new();
        for _ in 0..32 {
            threads.insert(format!(
                "{:?}",
                receiver
                    .recv_timeout(Duration::from_secs(5))
                    .expect("job completes")
            ));
        }
        assert!(threads.len() > 1, "expected work to spread across workers");
    }
}
