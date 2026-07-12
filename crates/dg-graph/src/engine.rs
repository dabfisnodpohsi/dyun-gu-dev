use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs;
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
use crate::registry::create_element;
use crate::spec::{ConnectionSpec, GraphSpec, NodeSpec, ParallelType};

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

/// A live execution of a graph. Workers and packet routes remain owned by
/// this handle until [`RunningGraph::finish`] joins them.
pub struct RunningGraph {
    spec: GraphSpec,
    stop: Arc<AtomicBool>,
    workers: BTreeMap<String, LiveNode>,
    routes: RuntimeRoutes,
    sinks: SinkMap,
    metrics: BTreeMap<String, Arc<ElementMetrics>>,
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
        self.start(inputs)?.finish()
    }

    /// Starts the graph without blocking the caller.
    pub fn start(&self, inputs: HashMap<String, Vec<Tensor>>) -> Result<RunningGraph> {
        let (runtime, sinks, metrics) = RuntimeGraph::build(self.spec.clone(), inputs)?;
        runtime.start(sinks, metrics)
    }
}

pub struct RuntimeGraph {
    nodes: Vec<ExecNode>,
    routes: RuntimeRoutes,
    spec: GraphSpec,
    stop: Arc<AtomicBool>,
}

struct RuntimeRoutes {
    edges: BTreeMap<String, EdgeRoute>,
    inputs: BTreeMap<(String, String), Arc<Mutex<PipeReceiver>>>,
    outputs: BTreeMap<(String, String), Arc<Mutex<Vec<PipeSender>>>>,
}

struct EdgeRoute {
    sender: PipeSender,
    receiver: Arc<Mutex<PipeReceiver>>,
}

struct LiveNode {
    control: Arc<crate::element::NodeControl>,
    workers: Vec<thread::JoinHandle<Result<()>>>,
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

        let mut edge_routes = BTreeMap::new();
        let mut input_routes = BTreeMap::new();
        let mut output_routes = BTreeMap::new();
        for connection in &spec.connections {
            let parsed = ConnectionSpec::parse(connection)?;
            let pipe = match spec.execution.parallel {
                ParallelType::Pipeline => DataPipe::bounded(spec.execution.queue_capacity),
                ParallelType::Sequential | ParallelType::Task => DataPipe::unbounded(),
            };
            let (sender, receiver) = pipe.split();
            let receiver = Arc::new(Mutex::new(receiver));
            {
                let src = nodes.get_mut(&parsed.from_node).ok_or_else(|| {
                    Error::Config(format!("missing source node {}", parsed.from_node))
                })?;
                src.outputs
                    .entry(parsed.from_port.clone())
                    .or_default()
                    .push(sender.clone());
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
            dst.inputs.insert(parsed.to_port.clone(), receiver.clone());
            edge_routes.insert(
                connection.clone(),
                EdgeRoute {
                    sender,
                    receiver: receiver.clone(),
                },
            );
        }

        for node in nodes.values() {
            for (port, receiver) in &node.inputs {
                input_routes.insert((node.name.clone(), port.clone()), receiver.clone());
            }
            for (port, senders) in &node.outputs {
                output_routes.insert(
                    (node.name.clone(), port.clone()),
                    Arc::new(Mutex::new(senders.clone())),
                );
            }
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
            let control = Arc::new(crate::element::NodeControl::default());
            for element in node.elements {
                let io = ElementIo {
                    name: node.name.clone(),
                    inputs: node
                        .inputs
                        .iter()
                        .map(|(port, receiver)| (port.clone(), receiver.clone()))
                        .collect(),
                    outputs: node
                        .outputs
                        .iter()
                        .map(|(port, senders)| {
                            (
                                port.clone(),
                                output_routes
                                    .get(&(node.name.clone(), port.clone()))
                                    .cloned()
                                    .unwrap_or_else(|| Arc::new(Mutex::new(senders.clone()))),
                            )
                        })
                        .collect(),
                    stop: stop.clone(),
                    control: control.clone(),
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
                routes: RuntimeRoutes {
                    edges: edge_routes,
                    inputs: input_routes,
                    outputs: output_routes,
                },
                spec: spec.clone(),
                stop,
            },
            sinks,
            metrics,
        ))
    }

    fn start(
        self,
        sinks: SinkMap,
        metrics: BTreeMap<String, Arc<ElementMetrics>>,
    ) -> Result<RunningGraph> {
        let mut workers = BTreeMap::new();
        let mut grouped: BTreeMap<String, Vec<ExecNode>> = BTreeMap::new();
        for node in self.nodes {
            grouped.entry(node.name.clone()).or_default().push(node);
        }
        for node_spec in &self.spec.nodes {
            let exec_nodes = grouped.remove(&node_spec.name).ok_or_else(|| {
                Error::Runtime(format!("missing runtime node {}", node_spec.name))
            })?;
            let first = exec_nodes.first().ok_or_else(|| {
                Error::Runtime(format!("node {} has no executable workers", node_spec.name))
            })?;
            let control = first.io.control.clone();
            let mut handles = Vec::with_capacity(exec_nodes.len());
            for node in exec_nodes {
                let stop = self.stop.clone();
                handles.push(thread::spawn(move || {
                    run_element(node.element, node.io, &stop)
                }));
            }
            workers.insert(
                node_spec.name.clone(),
                LiveNode {
                    control,
                    workers: handles,
                },
            );
        }
        if !grouped.is_empty() {
            return Err(Error::Runtime("runtime contains unknown nodes".to_string()));
        }
        Ok(RunningGraph {
            spec: self.spec,
            stop: self.stop,
            workers,
            routes: self.routes,
            sinks,
            metrics,
        })
    }
}

impl RunningGraph {
    /// Applies a validated graph diff while workers are running.
    pub fn apply_hot_update(&mut self, diff: GraphDiff) -> Result<()> {
        if diff.is_empty() {
            return Ok(());
        }
        let candidate = self.spec.clone().merge_for_diff(diff.clone())?;
        candidate.validate()?;

        let mut affected = BTreeMap::<String, ()>::new();
        for name in diff
            .removed_nodes
            .iter()
            .chain(diff.updated_nodes.iter().map(|node| &node.name))
        {
            affected.insert(name.clone(), ());
        }
        for node in &diff.added_nodes {
            affected.insert(node.name.clone(), ());
        }
        for connection in diff
            .added_connections
            .iter()
            .chain(diff.removed_connections.iter())
        {
            let parsed = ConnectionSpec::parse(connection)?;
            affected.insert(parsed.from_node, ());
            affected.insert(parsed.to_node, ());
        }

        let mut prepared = BTreeMap::new();
        for node in &candidate.nodes {
            if affected.contains_key(&node.name) {
                prepared.insert(node.name.clone(), PreparedNode::new(node)?);
            }
        }

        let affected_names = affected.keys().cloned().collect::<Vec<_>>();
        for name in &affected_names {
            if let Some(node) = self.workers.get(name) {
                node.control
                    .stop
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let mut next_edges = BTreeMap::new();
        let mut drain_routes = Vec::new();
        for connection in &candidate.connections {
            let parsed = ConnectionSpec::parse(connection)?;
            let old_route = self.routes.edges.remove(connection);
            let route = if !affected.contains_key(&parsed.to_node) {
                old_route.ok_or_else(|| {
                    Error::Runtime(format!("missing route for connection {connection}"))
                })?
            } else {
                let pipe = match candidate.execution.parallel {
                    ParallelType::Pipeline => DataPipe::bounded(candidate.execution.queue_capacity),
                    ParallelType::Sequential | ParallelType::Task => DataPipe::unbounded(),
                };
                let (sender, receiver) = pipe.split();
                if let Some(old_route) = old_route {
                    drain_routes.push((
                        old_route.receiver,
                        sender.clone(),
                        !affected.contains_key(&parsed.from_node),
                    ));
                }
                EdgeRoute {
                    sender,
                    receiver: Arc::new(Mutex::new(receiver)),
                }
            };
            next_edges.insert(connection.clone(), route);
        }

        let mut next_inputs = BTreeMap::new();
        let mut output_senders = BTreeMap::<(String, String), Vec<PipeSender>>::new();
        for connection in &candidate.connections {
            let parsed = ConnectionSpec::parse(connection)?;
            let route = next_edges.get(connection).ok_or_else(|| {
                Error::Runtime(format!("missing route for connection {connection}"))
            })?;
            next_inputs.insert(
                (parsed.to_node.clone(), parsed.to_port.clone()),
                route.receiver.clone(),
            );
            let output_key = (parsed.from_node.clone(), parsed.from_port.clone());
            output_senders
                .entry(output_key)
                .or_default()
                .push(route.sender.clone());
        }
        let mut next_outputs = BTreeMap::new();
        for (output_key, senders) in output_senders {
            let route = self
                .routes
                .outputs
                .get(&output_key)
                .cloned()
                .unwrap_or_else(|| Arc::new(Mutex::new(Vec::new())));
            *route
                .lock()
                .map_err(|_| Error::Runtime("output route lock poisoned".to_string()))? = senders;
            next_outputs.insert(output_key, route);
        }

        self.routes.edges = next_edges;
        self.routes.inputs = next_inputs.clone();
        self.routes.outputs = next_outputs;

        for name in &affected_names {
            if let Some(mut node) = self.workers.remove(name) {
                join_workers(&mut node.workers, true)?;
            }
        }

        for node in &candidate.nodes {
            if !affected.contains_key(&node.name) {
                continue;
            }
            let Some(prepared_node) = prepared.remove(&node.name) else {
                continue;
            };
            let control = Arc::new(crate::element::NodeControl::default());
            let eos = Arc::new(Mutex::new(EosState {
                seen: false,
                broadcasts: 0,
                instances: prepared_node.elements.len(),
            }));
            let node_metrics = Arc::new(ElementMetrics::default());
            let mut routes_in = HashMap::new();
            let mut routes_out = HashMap::new();
            for connection in &candidate.connections {
                let parsed = ConnectionSpec::parse(connection)?;
                if parsed.to_node == node.name {
                    let key = (node.name.clone(), parsed.to_port.clone());
                    let route = next_inputs
                        .get(&key)
                        .cloned()
                        .ok_or_else(|| Error::Runtime("missing input route".to_string()))?;
                    routes_in.insert(parsed.to_port, route);
                }
                if parsed.from_node == node.name {
                    let key = (node.name.clone(), parsed.from_port.clone());
                    let route = self
                        .routes
                        .outputs
                        .get(&key)
                        .cloned()
                        .ok_or_else(|| Error::Runtime("missing output route".to_string()))?;
                    routes_out.insert(parsed.from_port, route);
                }
            }
            let mut handles = Vec::with_capacity(prepared_node.elements.len());
            for element in prepared_node.elements {
                let io = ElementIo {
                    name: node.name.clone(),
                    inputs: routes_in.clone(),
                    outputs: routes_out.clone(),
                    stop: self.stop.clone(),
                    control: control.clone(),
                    send_backoff: Duration::from_millis(1),
                    eos: eos.clone(),
                    metrics: node_metrics.clone(),
                    packet_starts: std::cell::RefCell::new(VecDeque::new()),
                };
                let stop = self.stop.clone();
                handles.push(thread::spawn(move || run_element(element, io, &stop)));
            }
            self.workers.insert(
                node.name.clone(),
                LiveNode {
                    control,
                    workers: handles,
                },
            );
        }
        for (old_receiver, sender, upstream_stays_live) in drain_routes {
            loop {
                let packet = {
                    let receiver = old_receiver
                        .lock()
                        .map_err(|_| Error::Runtime("drain route lock poisoned".to_string()))?;
                    if upstream_stays_live {
                        receiver.recv_timeout(Duration::from_millis(1))
                    } else {
                        receiver.try_recv().map_err(|error| match error {
                            std::sync::mpsc::TryRecvError::Empty => {
                                std::sync::mpsc::RecvTimeoutError::Timeout
                            }
                            std::sync::mpsc::TryRecvError::Disconnected => {
                                std::sync::mpsc::RecvTimeoutError::Disconnected
                            }
                        })
                    }
                };
                match packet {
                    Ok(packet) => {
                        let mut pending = packet;
                        let is_eos = pending.is_eos();
                        loop {
                            match sender.try_send(pending) {
                                Ok(()) => break,
                                Err(std::sync::mpsc::TrySendError::Full(packet)) => {
                                    pending = packet;
                                    thread::sleep(Duration::from_millis(1));
                                    if self.stop.load(std::sync::atomic::Ordering::Relaxed) {
                                        return Err(Error::NotRunning);
                                    }
                                    continue;
                                }
                                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                                    return Err(Error::Runtime(
                                        "drain route disconnected".to_string(),
                                    ));
                                }
                            }
                        }
                        if is_eos {
                            break;
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) if !upstream_stays_live => {
                        break;
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        if self.stop.load(std::sync::atomic::Ordering::Relaxed) {
                            return Err(Error::NotRunning);
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
        }
        self.spec = candidate;
        Ok(())
    }

    /// Joins all workers and returns the collected sink report.
    pub fn finish(self) -> Result<GraphReport> {
        let mut first_error = None;
        for (_, mut node) in self.workers {
            if let Err(error) = join_workers(&mut node.workers, false) {
                first_error = select_error(first_error, error);
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        collect_report(&self.sinks, &self.metrics)
    }

    /// Alias for [`RunningGraph::finish`].
    pub fn join(self) -> Result<GraphReport> {
        self.finish()
    }
}

struct PreparedNode {
    elements: Vec<Box<dyn Element>>,
}

impl PreparedNode {
    fn new(node: &NodeSpec) -> Result<Self> {
        let threads = node.threads.unwrap_or(1);
        let created = create_element(node)?;
        if threads > 1 && (node.kind == "source" || !matches!(&created.handle, ElementHandle::None))
        {
            return Err(Error::Config(format!(
                "node {} cannot be multi-instanced because source elements and elements with special handles are single-instance",
                node.name
            )));
        }
        let mut elements = vec![created.element];
        for _ in 1..threads {
            let created = create_element(node)?;
            if node.kind == "source" || !matches!(&created.handle, ElementHandle::None) {
                return Err(Error::Config(format!(
                    "node {} cannot be multi-instanced because source elements and elements with special handles are single-instance",
                    node.name
                )));
            }
            elements.push(created.element);
        }
        Ok(Self { elements })
    }
}

fn join_workers(workers: &mut Vec<thread::JoinHandle<Result<()>>>, cancelled: bool) -> Result<()> {
    let mut first_error = None;
    while let Some(worker) = workers.pop() {
        match worker.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) if cancelled && is_cancellation(&error) => {}
            Ok(Err(error)) => first_error = select_error(first_error, error),
            Err(_) => {
                first_error = select_error(
                    first_error,
                    Error::Runtime("element worker panicked".to_string()),
                )
            }
        }
    }
    first_error.map_or(Ok(()), Err)
}

fn collect_report(
    sinks: &SinkMap,
    metrics: &BTreeMap<String, Arc<ElementMetrics>>,
) -> Result<GraphReport> {
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
            name.clone(),
            guard
                .ocr
                .iter()
                .flat_map(|batch| batch.iter().cloned())
                .collect(),
        );
    }
    for (node, metrics) in metrics {
        report
            .element_metrics
            .insert(node.clone(), metrics.snapshot());
    }
    Ok(report)
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

fn run_element(element: Box<dyn Element>, io: ElementIo, stop: &Arc<AtomicBool>) -> Result<()> {
    match catch_unwind(AssertUnwindSafe(|| element.run(io))) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => {
            if !is_cancellation(&err) {
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            Err(err)
        }
        Err(_) => {
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            Err(Error::Runtime("element panicked".to_string()))
        }
    }
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
