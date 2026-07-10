use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{self, sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use dg_core::{Classification, Detection, FaceDetection, OcrText, Tensor, Track};
use tracing::{error, info};

use crate::element::{Element, ElementHandle, ElementIo};
use crate::error::{Error, Result};
use crate::packet::Packet;
use crate::registry::create_element;
use crate::spec::{ConnectionSpec, GraphSpec, NodeSpec};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphDiff {
    pub added_nodes: Vec<NodeSpec>,
    pub removed_nodes: Vec<String>,
    pub updated_nodes: Vec<NodeSpec>,
    pub added_connections: Vec<String>,
    pub removed_connections: Vec<String>,
}

impl GraphDiff {
    pub fn is_empty(&self) -> bool {
        self.added_nodes.is_empty()
            && self.removed_nodes.is_empty()
            && self.updated_nodes.is_empty()
            && self.added_connections.is_empty()
            && self.removed_connections.is_empty()
    }

    pub fn apply(self, graph: &mut Graph) -> Result<()> {
        let new_spec = graph.spec.clone().merge_for_diff(self)?;
        *graph = Graph::new(new_spec)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct GraphReport {
    pub sinks: BTreeMap<String, Vec<Tensor>>,
    pub detections: BTreeMap<String, Vec<Detection>>,
    pub classifications: BTreeMap<String, Vec<Classification>>,
    pub faces: BTreeMap<String, Vec<FaceDetection>>,
    pub tracks: BTreeMap<String, Vec<Track>>,
    pub ocr: BTreeMap<String, Vec<OcrText>>,
}

type SinkMap = BTreeMap<String, Arc<Mutex<crate::element::SinkCollector>>>;

pub struct Graph {
    spec: GraphSpec,
}

impl Graph {
    pub fn new(spec: GraphSpec) -> Result<Self> {
        spec.validate()?;
        Ok(Self { spec })
    }

    pub fn spec(&self) -> &GraphSpec {
        &self.spec
    }

    pub fn diff(old: &GraphSpec, new: &GraphSpec) -> GraphDiff {
        let old_nodes: BTreeMap<_, _> = old
            .nodes
            .iter()
            .map(|node| (node.name.clone(), node.clone()))
            .collect();
        let new_nodes: BTreeMap<_, _> = new
            .nodes
            .iter()
            .map(|node| (node.name.clone(), node.clone()))
            .collect();

        let mut added_nodes = Vec::new();
        let mut removed_nodes = Vec::new();
        let mut updated_nodes = Vec::new();
        for (name, node) in &new_nodes {
            match old_nodes.get(name) {
                None => added_nodes.push(node.clone()),
                Some(existing) if existing != node => updated_nodes.push(node.clone()),
                Some(_) => {}
            }
        }
        for name in old_nodes.keys() {
            if !new_nodes.contains_key(name) {
                removed_nodes.push(name.clone());
            }
        }

        let old_connections = old
            .connections
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let new_connections = new
            .connections
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let added_connections = new_connections
            .difference(&old_connections)
            .cloned()
            .collect();
        let removed_connections = old_connections
            .difference(&new_connections)
            .cloned()
            .collect();

        GraphDiff {
            added_nodes,
            removed_nodes,
            updated_nodes,
            added_connections,
            removed_connections,
        }
    }

    pub fn reload(&mut self, spec: GraphSpec) -> Result<GraphDiff> {
        let diff = Self::diff(&self.spec, &spec);
        spec.validate()?;
        self.spec = spec;
        Ok(diff)
    }

    pub fn run(&self) -> Result<GraphReport> {
        self.run_with_inputs(HashMap::new())
    }

    pub fn run_with_inputs(&self, inputs: HashMap<String, Vec<Tensor>>) -> Result<GraphReport> {
        info!(
            node_count = self.spec.nodes.len(),
            "starting graph execution"
        );
        let (runtime, sinks) = RuntimeGraph::build(self.spec.clone(), inputs)?;
        runtime.run()?;
        let mut report = GraphReport::default();
        for (name, sink) in sinks {
            let guard = sink
                .lock()
                .map_err(|_| Error::Runtime("sink lock poisoned".to_string()))?;
            report.sinks.insert(name.clone(), guard.tensors.clone());
            report.detections.insert(
                name.clone(),
                guard
                    .detections
                    .iter()
                    .flat_map(|batch| batch.iter().cloned())
                    .collect(),
            );
            report.classifications.insert(
                name.clone(),
                guard
                    .classifications
                    .iter()
                    .flat_map(|batch| batch.iter().cloned())
                    .collect(),
            );
            report.faces.insert(
                name.clone(),
                guard
                    .faces
                    .iter()
                    .flat_map(|batch| batch.iter().cloned())
                    .collect(),
            );
            report.tracks.insert(
                name.clone(),
                guard
                    .tracks
                    .iter()
                    .flat_map(|batch| batch.iter().cloned())
                    .collect(),
            );
            report.ocr.insert(
                name,
                guard
                    .ocr
                    .iter()
                    .flat_map(|batch| batch.iter().cloned())
                    .collect(),
            );
        }
        Ok(report)
    }
}

pub struct RuntimeGraph {
    threads: Vec<thread::JoinHandle<Result<()>>>,
}

impl RuntimeGraph {
    fn build(spec: GraphSpec, inputs: HashMap<String, Vec<Tensor>>) -> Result<(Self, SinkMap)> {
        let stop = Arc::new(AtomicBool::new(false));
        let mut nodes: BTreeMap<String, NodeRuntime> = BTreeMap::new();
        for node in &spec.nodes {
            let created = create_element(node)?;
            nodes.insert(
                node.name.clone(),
                NodeRuntime {
                    name: node.name.clone(),
                    element: Some(created.element),
                    handle: created.handle,
                    inputs: HashMap::new(),
                    outputs: HashMap::new(),
                },
            );
        }

        let mut sinks = BTreeMap::new();
        let mut input_queues = BTreeMap::new();
        for (name, node) in &mut nodes {
            if let ElementHandle::Sink(collector) = &node.handle {
                sinks.insert(name.clone(), collector.clone());
            } else if let ElementHandle::Input(queue) = &node.handle {
                input_queues.insert(name.clone(), queue.clone());
            }
        }

        for (name, tensors) in inputs {
            let queue = input_queues.get(&name).ok_or_else(|| {
                Error::Config(format!("unknown input node {} for injected tensors", name))
            })?;
            let mut guard = queue
                .lock()
                .map_err(|_| Error::Runtime("input queue poisoned".to_string()))?;
            guard.extend(tensors);
        }

        for connection in &spec.connections {
            let parsed = ConnectionSpec::parse(connection)?;
            let pipe = DataPipe::bounded(20);
            let (sender, receiver) = pipe.split();
            {
                let src = nodes.get_mut(&parsed.from_node).ok_or_else(|| {
                    Error::Config(format!("missing source node {}", parsed.from_node))
                })?;
                src.outputs
                    .entry(parsed.from_port.clone())
                    .or_default()
                    .push(sender);
            }
            let dst = nodes.get_mut(&parsed.to_node).ok_or_else(|| {
                Error::Config(format!("missing destination node {}", parsed.to_node))
            })?;
            if dst.inputs.contains_key(&parsed.to_port) {
                return Err(Error::Config(format!(
                    "multiple inbound edges to {}.{} are not supported",
                    parsed.to_node, parsed.to_port
                )));
            }
            dst.inputs.insert(parsed.to_port.clone(), receiver);
        }

        for node in nodes.values() {
            for port in node.inputs.keys() {
                if !spec.connections.iter().any(|conn| {
                    ConnectionSpec::parse(conn).ok().is_some_and(|parsed| {
                        parsed.to_node == node.name && parsed.to_port == *port
                    })
                }) {
                    return Err(Error::Config(format!(
                        "input port {}.{} has no upstream connection",
                        node.name, port
                    )));
                }
            }
        }

        let mut threads = Vec::with_capacity(nodes.len());
        for mut node in nodes.into_values() {
            let stop = stop.clone();
            let backoff = Duration::from_millis(1);
            let io = ElementIo {
                name: node.name.clone(),
                inputs: node.inputs.drain().collect(),
                outputs: node.outputs.drain().collect(),
                stop: stop.clone(),
                send_backoff: backoff,
            };
            let element = node
                .element
                .take()
                .ok_or_else(|| Error::Config("missing element".to_string()))?;
            let thread_stop = stop.clone();
            threads.push(thread::spawn(move || {
                let result = catch_unwind(AssertUnwindSafe(|| element.run(io)));
                match result {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(err)) => {
                        thread_stop.store(true, std::sync::atomic::Ordering::Relaxed);
                        Err(err)
                    }
                    Err(_) => {
                        thread_stop.store(true, std::sync::atomic::Ordering::Relaxed);
                        Err(Error::Runtime("element panicked".to_string()))
                    }
                }
            }));
        }

        Ok((Self { threads }, sinks))
    }

    fn run(self) -> Result<()> {
        let mut first_error = None;
        for thread in self.threads {
            match thread.join() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
                Err(_) => {
                    if first_error.is_none() {
                        first_error = Some(Error::Runtime("element panicked".to_string()));
                    }
                }
            }
        }
        match first_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

struct NodeRuntime {
    name: String,
    element: Option<Box<dyn Element>>,
    handle: ElementHandle,
    inputs: HashMap<String, Receiver<Packet>>,
    outputs: HashMap<String, Vec<SyncSender<Packet>>>,
}

pub struct DataPipe {
    sender: SyncSender<Packet>,
    receiver: Receiver<Packet>,
}

impl DataPipe {
    pub fn bounded(capacity: usize) -> Self {
        let (sender, receiver) = sync_channel(capacity);
        Self { sender, receiver }
    }

    pub fn split(self) -> (SyncSender<Packet>, Receiver<Packet>) {
        (self.sender, self.receiver)
    }
}

impl GraphSpec {
    fn merge_for_diff(self, diff: GraphDiff) -> Result<Self> {
        if diff.is_empty() {
            return Ok(self);
        }
        let mut spec = self;
        for node in diff.removed_nodes {
            spec.nodes.retain(|existing| existing.name != node);
        }
        for node in diff.added_nodes {
            spec.nodes.push(node);
        }
        for node in diff.updated_nodes {
            spec.nodes.retain(|existing| existing.name != node.name);
            spec.nodes.push(node);
        }
        for conn in diff.removed_connections {
            spec.connections.retain(|existing| existing != &conn);
        }
        spec.connections.extend(diff.added_connections);
        Ok(spec)
    }
}

/// Controls a background graph specification file watcher.
pub struct WatchHandle {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl WatchHandle {
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for WatchHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Watches a graph specification file and reports validated changes.
pub fn watch(
    path: impl AsRef<Path>,
    mut callback: impl FnMut(Result<(GraphSpec, GraphDiff)>) + Send + 'static,
) -> Result<WatchHandle> {
    let path = path.as_ref().to_path_buf();
    let mut modified = fs::metadata(&path)?.modified()?;
    let mut previous = GraphSpec::load_from_path(&path)?;
    let (stop, stop_receiver) = mpsc::channel();
    let thread = thread::spawn(move || {
        const POLL_INTERVAL: Duration = Duration::from_millis(50);
        loop {
            match stop_receiver.recv_timeout(POLL_INTERVAL) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }

            let metadata = match fs::metadata(&path) {
                Ok(metadata) => metadata,
                Err(error) => {
                    notify_watch(&mut callback, Err(error.into()));
                    continue;
                }
            };
            let current_modified = match metadata.modified() {
                Ok(modified) => modified,
                Err(error) => {
                    notify_watch(&mut callback, Err(error.into()));
                    continue;
                }
            };
            if current_modified == modified {
                continue;
            }

            match GraphSpec::load_from_path(&path) {
                Ok(spec) => {
                    let diff = Graph::diff(&previous, &spec);
                    previous = spec.clone();
                    modified = current_modified;
                    notify_watch(&mut callback, Ok((spec, diff)));
                }
                Err(error) => {
                    modified = current_modified;
                    error!(path = %path.display(), error = %error, "graph watch reload failed");
                    notify_watch(&mut callback, Err(error));
                }
            }
        }
    });
    Ok(WatchHandle {
        stop: Some(stop),
        thread: Some(thread),
    })
}

fn notify_watch(
    callback: &mut impl FnMut(Result<(GraphSpec, GraphDiff)>),
    result: Result<(GraphSpec, GraphDiff)>,
) {
    if catch_unwind(AssertUnwindSafe(|| callback(result))).is_err() {
        error!("graph watch callback panicked");
    }
}
