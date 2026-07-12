use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs;
use std::num::NonZeroUsize;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use dg_core::{Classification, Detection, FaceDetection, OcrText, Tensor, Track};
use tracing::{error, info};

use crate::element::{Element, ElementHandle, ElementIo, EosState};
use crate::error::{Error, Result};
use crate::metrics::{ElementMetrics, ElementMetricsSnapshot, MetricsSink};
use crate::pipe::{DataPipe, PipeReceiver, PipeSender};
use crate::pool::ThreadPool;
use crate::registry::create_element;
use crate::spec::{ConnectionSpec, ExecutionSpec, GraphSpec, NodeSpec, ParallelType};

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
        graph.reload(new_spec)?;
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
    pub element_metrics: BTreeMap<String, ElementMetricsSnapshot>,
}

impl GraphReport {
    pub fn export_metrics(&self, sink: &dyn MetricsSink) {
        for (node, metrics) in &self.element_metrics {
            sink.record(node, metrics);
        }
    }
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
            parallel = ?self.spec.execution.parallel,
            queue_capacity = self.spec.execution.queue_capacity,
            "starting graph execution"
        );
        let (runtime, sinks, metrics) = RuntimeGraph::build(self.spec.clone(), inputs)?;
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
        for (node, metrics) in metrics {
            let snapshot = metrics.snapshot();
            info!(
                node = %node,
                packets_processed = snapshot.packets_processed,
                packets_received = snapshot.packets_received,
                packets_sent = snapshot.packets_sent,
                processing_latency_ns = snapshot.processing_latency_ns,
                processing_latency_avg_ns = snapshot.processing_latency_avg_ns,
                processing_latency_max_ns = snapshot.processing_latency_max_ns,
                queue_depth = snapshot.queue_depth,
                max_queue_depth = snapshot.max_queue_depth,
                drop_count = snapshot.drop_count,
                backpressure_count = snapshot.backpressure_count,
                "element metrics"
            );
            report.element_metrics.insert(node, snapshot);
        }
        Ok(report)
    }
}

pub struct RuntimeGraph {
    nodes: Vec<ExecNode>,
    edges: Vec<(String, String)>,
    stop: Arc<AtomicBool>,
    execution: ExecutionSpec,
}

impl RuntimeGraph {
    fn build(
        spec: GraphSpec,
        inputs: HashMap<String, Vec<Tensor>>,
    ) -> Result<(Self, SinkMap, BTreeMap<String, Arc<ElementMetrics>>)> {
        let stop = Arc::new(AtomicBool::new(false));
        let mut nodes: BTreeMap<String, NodeRuntime> = BTreeMap::new();
        for node in &spec.nodes {
            let threads = node.threads.unwrap_or(1);
            let created = create_element(node)?;
            if threads > 1
                && (node.kind == "source" || !matches!(&created.handle, ElementHandle::None))
            {
                return Err(Error::Config(format!(
                    "node {} cannot be multi-instanced because source elements and elements with special handles are single-instance",
                    node.name,
                )));
            }
            let handle = created.handle;
            let mut elements = vec![created.element];
            for _ in 1..threads {
                let created = create_element(node)?;
                if node.kind == "source" || !matches!(&created.handle, ElementHandle::None) {
                    return Err(Error::Config(format!(
                        "node {} cannot be multi-instanced because source elements and elements with special handles are single-instance",
                        node.name,
                    )));
                }
                elements.push(created.element);
            }
            nodes.insert(
                node.name.clone(),
                NodeRuntime {
                    name: node.name.clone(),
                    elements,
                    handle,
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

        let mut edges = Vec::with_capacity(spec.connections.len());
        for connection in &spec.connections {
            let parsed = ConnectionSpec::parse(connection)?;
            edges.push((parsed.from_node.clone(), parsed.to_node.clone()));
            let pipe = match spec.execution.parallel {
                ParallelType::Pipeline => DataPipe::bounded(spec.execution.queue_capacity),
                ParallelType::Sequential | ParallelType::Task => DataPipe::unbounded(),
            };
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
            dst.inputs
                .insert(parsed.to_port.clone(), Arc::new(Mutex::new(receiver)));
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

        let total_elements = nodes.values().map(|node| node.elements.len()).sum();
        let mut exec_nodes = Vec::with_capacity(total_elements);
        let mut metrics = BTreeMap::new();
        for node in nodes.into_values() {
            let node_metrics = Arc::new(ElementMetrics::default());
            metrics.insert(node.name.clone(), node_metrics.clone());
            let eos = Arc::new(Mutex::new(EosState {
                seen: false,
                broadcasts: 0,
                instances: node.elements.len(),
            }));
            for element in node.elements {
                let io = ElementIo {
                    name: node.name.clone(),
                    inputs: node
                        .inputs
                        .iter()
                        .map(|(port, receiver)| (port.clone(), receiver.clone()))
                        .collect(),
                    outputs: node.outputs.clone(),
                    stop: stop.clone(),
                    send_backoff: Duration::from_millis(1),
                    eos: eos.clone(),
                    metrics: node_metrics.clone(),
                    packet_starts: std::cell::RefCell::new(VecDeque::new()),
                };
                exec_nodes.push(ExecNode {
                    name: node.name.clone(),
                    element,
                    io,
                });
            }
        }

        Ok((
            Self {
                nodes: exec_nodes,
                edges,
                stop,
                execution: spec.execution.clone(),
            },
            sinks,
            metrics,
        ))
    }

    fn run(self) -> Result<()> {
        match self.execution.parallel {
            ParallelType::Sequential => self.run_sequential(),
            ParallelType::Task => self.run_task(),
            ParallelType::Pipeline => self.run_pipeline(),
        }
    }

    fn run_sequential(self) -> Result<()> {
        let order = topological_order(&self.nodes, &self.edges)?;
        let mut by_name: HashMap<String, ExecNode> = self
            .nodes
            .into_iter()
            .map(|node| (node.name.clone(), node))
            .collect();
        for name in order {
            let node = by_name
                .remove(&name)
                .ok_or_else(|| Error::Runtime(format!("missing runtime node {name}")))?;
            run_element(node.element, node.io, &self.stop)?;
        }
        Ok(())
    }

    fn run_pipeline(self) -> Result<()> {
        let pool = ThreadPool::new(self.nodes.len().max(1))?;
        let (results, receiver) = mpsc::channel();
        let total = self.nodes.len();
        for node in self.nodes {
            let results = results.clone();
            let stop = self.stop.clone();
            pool.spawn(move || {
                let result = run_element(node.element, node.io, &stop);
                let _ = results.send(result);
            })?;
        }
        drop(results);
        let mut first_error = None;
        for _ in 0..total {
            match receiver.recv() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    first_error = select_error(first_error, err);
                }
                Err(_) => {
                    first_error = select_error(
                        first_error,
                        Error::Runtime("element worker lost".to_string()),
                    );
                    break;
                }
            }
        }
        match first_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    fn run_task(self) -> Result<()> {
        // Reject cycles up front; dependency-driven scheduling cannot make
        // progress on cyclic graphs.
        topological_order(&self.nodes, &self.edges)?;
        let workers = match self.execution.workers {
            Some(workers) => workers,
            None => thread::available_parallelism()
                .map(NonZeroUsize::get)
                .unwrap_or(1),
        }
        .min(self.nodes.len().max(1));
        let pool = ThreadPool::new(workers)?;

        let mut indegree: HashMap<String, usize> = self
            .nodes
            .iter()
            .map(|node| (node.name.clone(), 0))
            .collect();
        let mut dependents: HashMap<String, Vec<String>> = HashMap::new();
        for (from, to) in &self.edges {
            if let Some(count) = indegree.get_mut(to) {
                *count += 1;
            }
            dependents.entry(from.clone()).or_default().push(to.clone());
        }

        let mut waiting: HashMap<String, ExecNode> = self
            .nodes
            .into_iter()
            .map(|node| (node.name.clone(), node))
            .collect();
        let ready: Vec<String> = indegree
            .iter()
            .filter(|(_, count)| **count == 0)
            .map(|(name, _)| name.clone())
            .collect();

        let (results, receiver) = mpsc::channel::<(String, Result<()>)>();
        let mut running = 0usize;
        for name in ready {
            spawn_task(&pool, &mut waiting, &name, &self.stop, &results)?;
            running += 1;
        }

        let mut first_error = None;
        while running > 0 {
            let (name, result) = receiver
                .recv()
                .map_err(|_| Error::Runtime("element worker lost".to_string()))?;
            running -= 1;
            match result {
                Ok(()) => {
                    if first_error.is_none() {
                        for dependent in dependents.remove(&name).unwrap_or_default() {
                            let Some(count) = indegree.get_mut(&dependent) else {
                                continue;
                            };
                            *count -= 1;
                            if *count == 0 {
                                spawn_task(&pool, &mut waiting, &dependent, &self.stop, &results)?;
                                running += 1;
                            }
                        }
                    }
                }
                Err(err) => {
                    first_error = select_error(first_error, err);
                }
            }
        }
        match first_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

fn is_cancellation(error: &Error) -> bool {
    matches!(error, Error::NotRunning)
}

fn select_error(current: Option<Error>, candidate: Error) -> Option<Error> {
    match current {
        Some(existing) if !is_cancellation(&existing) || is_cancellation(&candidate) => {
            Some(existing)
        }
        _ => Some(candidate),
    }
}

fn spawn_task(
    pool: &ThreadPool,
    waiting: &mut HashMap<String, ExecNode>,
    name: &str,
    stop: &Arc<AtomicBool>,
    results: &mpsc::Sender<(String, Result<()>)>,
) -> Result<()> {
    let node = waiting
        .remove(name)
        .ok_or_else(|| Error::Runtime(format!("missing runtime node {name}")))?;
    let stop = stop.clone();
    let results = results.clone();
    pool.spawn(move || {
        let result = run_element(node.element, node.io, &stop);
        let _ = results.send((node.name, result));
    })
}

fn run_element(element: Box<dyn Element>, io: ElementIo, stop: &Arc<AtomicBool>) -> Result<()> {
    match catch_unwind(AssertUnwindSafe(|| element.run(io))) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => {
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            Err(err)
        }
        Err(_) => {
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            Err(Error::Runtime("element panicked".to_string()))
        }
    }
}

fn topological_order(nodes: &[ExecNode], edges: &[(String, String)]) -> Result<Vec<String>> {
    let mut indegree: BTreeMap<&str, usize> =
        nodes.iter().map(|node| (node.name.as_str(), 0)).collect();
    let mut adjacency: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (from, to) in edges {
        if let Some(count) = indegree.get_mut(to.as_str()) {
            *count += 1;
        }
        adjacency.entry(from.as_str()).or_default().push(to);
    }
    let mut ready: Vec<&str> = indegree
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(name, _)| *name)
        .collect();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(name) = ready.pop() {
        order.push(name.to_string());
        for &next in adjacency.get(name).map(Vec::as_slice).unwrap_or_default() {
            if let Some(count) = indegree.get_mut(next) {
                *count -= 1;
                if *count == 0 {
                    ready.push(next);
                }
            }
        }
    }
    if order.len() != nodes.len() {
        return Err(Error::Config(
            "sequential/task execution requires an acyclic graph".to_string(),
        ));
    }
    Ok(order)
}

struct NodeRuntime {
    name: String,
    elements: Vec<Box<dyn Element>>,
    handle: ElementHandle,
    inputs: HashMap<String, Arc<Mutex<PipeReceiver>>>,
    outputs: HashMap<String, Vec<PipeSender>>,
}

struct ExecNode {
    name: String,
    element: Box<dyn Element>,
    io: ElementIo,
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
#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use dg_core::DataType;
    use serde_json::json;

    use super::*;
    use crate::element::{CreatedElement, PortSchema};
    use crate::registry::ElementDescriptor;
    use crate::spec::{GraphSpecBuilder, NodeSpec};

    static THREADED_INSTANCE_COUNT: AtomicUsize = AtomicUsize::new(0);

    const TEST_INPUT: PortSchema = PortSchema {
        name: "in",
        dtype: Some(DataType::F32),
        required: true,
    };
    const TEST_OUTPUT: PortSchema = PortSchema {
        name: "out",
        dtype: Some(DataType::F32),
        required: false,
    };

    struct ThreadedPassthrough;

    impl Element for ThreadedPassthrough {
        fn run(self: Box<Self>, io: ElementIo) -> Result<()> {
            loop {
                let packet = match io.recv("in")? {
                    Some(packet) => packet,
                    None => continue,
                };
                if packet.is_eos() {
                    io.broadcast_eos()?;
                    return Ok(());
                }
                io.send("out", packet)?;
            }
        }
    }

    fn create_threaded_passthrough(_: &NodeSpec) -> Result<CreatedElement> {
        THREADED_INSTANCE_COUNT.fetch_add(1, Ordering::SeqCst);
        Ok(CreatedElement {
            element: Box::new(ThreadedPassthrough),
            handle: ElementHandle::None,
        })
    }

    inventory::submit! {
        ElementDescriptor {
            kind: "threaded_test_passthrough",
            input_ports: &[TEST_INPUT],
            output_ports: &[TEST_OUTPUT],
            params: &[],
            validate: None,
            create: create_threaded_passthrough,
        }
    }

    #[test]
    fn pipeline_creates_and_runs_each_requested_instance() {
        THREADED_INSTANCE_COUNT.store(0, Ordering::SeqCst);
        let spec = GraphSpecBuilder::new()
            .add_node(NodeSpec {
                name: "source".to_string(),
                kind: "source".to_string(),
                params: json!({"count": 8, "shape": [1, 4]}),
                ..NodeSpec::default()
            })
            .add_node(NodeSpec {
                name: "threaded".to_string(),
                kind: "threaded_test_passthrough".to_string(),
                threads: Some(2),
                params: json!({}),
                ..NodeSpec::default()
            })
            .add_node(NodeSpec {
                name: "sink".to_string(),
                kind: "sink".to_string(),
                params: json!({}),
                ..NodeSpec::default()
            })
            .connect("source.out -> threaded.in")
            .connect("threaded.out -> sink.in")
            .build()
            .expect("build threaded test graph");

        let report = Graph::new(spec)
            .expect("construct threaded test graph")
            .run()
            .expect("run threaded test graph");
        assert_eq!(
            THREADED_INSTANCE_COUNT.load(Ordering::SeqCst),
            2,
            "requested instances should each be created"
        );
        assert_eq!(report.sinks["sink"].len(), 8);
    }

    fn root_cause() -> Error {
        Error::Element {
            element: "decode".to_string(),
            message: "recorded frame has an invalid payload size".to_string(),
        }
    }

    #[test]
    fn error_selection_prefers_root_cause_over_cancellation() {
        let selected = select_error(Some(Error::NotRunning), root_cause());
        assert!(matches!(selected, Some(Error::Element { .. })));
    }

    #[test]
    fn error_selection_keeps_root_cause_when_cancellation_arrives_later() {
        let selected = select_error(Some(root_cause()), Error::NotRunning);
        assert!(matches!(selected, Some(Error::Element { .. })));
    }

    #[test]
    fn cancellation_is_only_the_not_running_error() {
        assert!(is_cancellation(&Error::NotRunning));
        assert!(!is_cancellation(&root_cause()));
    }
}
