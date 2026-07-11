use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::element::PortSchema;
use crate::error::{Error, Result};
use crate::pipe::DEFAULT_QUEUE_CAPACITY;
use crate::registry::{element_ports, find_element, validate_element};

const DEFAULT_API_VERSION: &str = "dg/v1";
const DEFAULT_KIND: &str = "Graph";

/// How graph elements are scheduled onto threads (from nndeploy).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ParallelType {
    /// Elements run one at a time in topological order on the calling thread.
    Sequential,
    /// Elements run as dataflow tasks on a work-stealing pool once their
    /// upstream elements complete.
    Task,
    /// Every element gets a dedicated pool thread; bounded pipes apply
    /// backpressure between concurrently running elements.
    #[default]
    Pipeline,
}

/// Execution parameters for a graph.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct ExecutionSpec {
    pub parallel: ParallelType,
    /// Capacity of each bounded `DataPipe` in pipeline mode.
    pub queue_capacity: usize,
    /// Worker count for task mode; defaults to available parallelism.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workers: Option<usize>,
}

impl Default for ExecutionSpec {
    fn default() -> Self {
        Self {
            parallel: ParallelType::default(),
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
            workers: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NodeTemplate {
    pub kind: String,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub params: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum DeviceDefault {
    Named(String),
    Detailed(DeviceDefaultDetails),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeviceDefaultDetails {
    pub kind: String,
    #[serde(default)]
    pub id: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct DefaultsSpec {
    pub backend: Option<String>,
    pub device: Option<DeviceDefault>,
    pub precision: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NodeSpec {
    pub name: String,
    #[serde(alias = "type")]
    pub kind: String,
    /// Number of Pipeline instances for this node.
    #[serde(default)]
    pub threads: Option<usize>,
    /// Marks this node as terminal; terminal nodes cannot have outgoing edges.
    #[serde(default)]
    pub sink: bool,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub precision: Option<String>,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub params: Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConnectionSpec {
    pub from_node: String,
    pub from_port: String,
    pub to_node: String,
    pub to_port: String,
}

impl ConnectionSpec {
    pub fn parse(spec: &str) -> Result<Self> {
        let (from, to) = spec
            .split_once("->")
            .ok_or_else(|| Error::Config(format!("invalid connection: {spec}")))?;
        let from = from.trim();
        let to = to.trim();
        let (from_node, from_port) = from
            .split_once('.')
            .ok_or_else(|| Error::Config(format!("invalid source endpoint: {from}")))?;
        let (to_node, to_port) = to
            .split_once('.')
            .ok_or_else(|| Error::Config(format!("invalid destination endpoint: {to}")))?;
        Ok(Self {
            from_node: from_node.trim().to_string(),
            from_port: from_port.trim().to_string(),
            to_node: to_node.trim().to_string(),
            to_port: to_port.trim().to_string(),
        })
    }
}

impl Display for ConnectionSpec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{} -> {}.{}",
            self.from_node, self.from_port, self.to_node, self.to_port
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GraphSpec {
    #[serde(rename = "apiVersion", default = "default_api_version")]
    pub api_version: String,
    #[serde(default = "default_kind")]
    pub kind: String,
    #[serde(default)]
    pub includes: Vec<String>,
    #[serde(default)]
    #[serde(alias = "vars")]
    pub variables: BTreeMap<String, Value>,
    #[serde(default)]
    pub defaults: DefaultsSpec,
    #[serde(default)]
    pub templates: BTreeMap<String, NodeTemplate>,
    #[serde(default)]
    pub allow_cycles: bool,
    #[serde(default)]
    pub execution: ExecutionSpec,
    #[serde(default)]
    pub nodes: Vec<NodeSpec>,
    #[serde(default)]
    #[serde(alias = "edges")]
    pub connections: Vec<String>,
}

fn default_api_version() -> String {
    DEFAULT_API_VERSION.to_string()
}

fn default_kind() -> String {
    DEFAULT_KIND.to_string()
}

impl Default for GraphSpec {
    fn default() -> Self {
        Self {
            api_version: default_api_version(),
            kind: default_kind(),
            includes: Vec::new(),
            variables: BTreeMap::new(),
            defaults: DefaultsSpec::default(),
            templates: BTreeMap::new(),
            allow_cycles: false,
            execution: ExecutionSpec::default(),
            nodes: Vec::new(),
            connections: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphFormat {
    Yaml,
    Json,
    Toml,
}

impl GraphFormat {
    pub fn from_path(path: &Path) -> Result<Self> {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("yaml") | Some("yml") => Ok(Self::Yaml),
            Some("json") => Ok(Self::Json),
            Some("toml") => Ok(Self::Toml),
            _ => Err(Error::UnknownFormat(path.to_path_buf())),
        }
    }
}

impl GraphSpec {
    /// Exports the JSON Schema describing the configuration model.
    pub fn json_schema() -> Result<String> {
        Ok(serde_json::to_string_pretty(&schema_for!(GraphSpec))?)
    }

    pub fn from_str_with_format(input: &str, format: GraphFormat) -> Result<Self> {
        let spec = match format {
            GraphFormat::Yaml => serde_yaml_ng::from_str(input)?,
            GraphFormat::Json => serde_json::from_str(input)?,
            GraphFormat::Toml => toml::from_str(input)?,
        };
        Ok(spec)
    }

    pub fn to_string_with_format(&self, format: GraphFormat) -> Result<String> {
        match format {
            GraphFormat::Yaml => Ok(serde_yaml_ng::to_string(self)?),
            GraphFormat::Json => Ok(serde_json::to_string_pretty(self)?),
            GraphFormat::Toml => Ok(toml::to_string_pretty(self)?),
        }
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let mut resolving = BTreeSet::new();
        Self::load_from_path_tracked(path.as_ref(), &mut resolving)
    }

    fn load_from_path_tracked(path: &Path, resolving: &mut BTreeSet<PathBuf>) -> Result<Self> {
        let canonical_path = fs::canonicalize(path)?;
        if !resolving.insert(canonical_path.clone()) {
            return Err(Error::Validation {
                path: "includes".to_string(),
                message: format!("include cycle detected at {}", path.display()),
            });
        }
        let result = Self::load_from_path_tracked_inner(path, resolving);
        resolving.remove(&canonical_path);
        result
    }

    fn load_from_path_tracked_inner(
        path: &Path,
        resolving: &mut BTreeSet<PathBuf>,
    ) -> Result<Self> {
        let format = GraphFormat::from_path(path)?;
        let content = fs::read_to_string(path)?;
        let spec = Self::from_str_with_format(&content, format)?;
        spec.normalize_with_base_dir_tracked(path.parent(), resolving)
    }

    pub fn normalize_with_base_dir(self, base_dir: Option<&Path>) -> Result<Self> {
        let mut resolving = BTreeSet::new();
        self.normalize_with_base_dir_tracked(base_dir, &mut resolving)
    }

    fn normalize_with_base_dir_tracked(
        self,
        base_dir: Option<&Path>,
        resolving: &mut BTreeSet<PathBuf>,
    ) -> Result<Self> {
        if !(self.api_version == "dg/v1" || self.api_version == "v1") {
            return Err(Error::Validation {
                path: "apiVersion".to_string(),
                message: format!("unsupported apiVersion: {}", self.api_version),
            });
        }
        if self.kind != DEFAULT_KIND {
            return Err(Error::Validation {
                path: "kind".to_string(),
                message: format!("unsupported kind: {}", self.kind),
            });
        }
        if base_dir.is_none() && !self.includes.is_empty() {
            return Err(Error::Validation {
                path: "includes".to_string(),
                message: "includes require loading from a file path with a base directory"
                    .to_string(),
            });
        }

        let mut merged = GraphSpec::default();
        if let Some(base_dir) = base_dir {
            for include in &self.includes {
                let included_path = base_dir.join(include);
                let included = GraphSpec::load_from_path_tracked(&included_path, resolving)?;
                merged.merge_included(included);
            }
        }

        merged.merge_included(self.clone());
        merged.includes.clear();
        let explicit_param_keys = merged
            .nodes
            .iter()
            .map(|node| match &node.params {
                Value::Object(params) => params.keys().cloned().collect(),
                _ => BTreeSet::new(),
            })
            .collect::<Vec<BTreeSet<String>>>();
        merged.apply_templates()?;
        merged.apply_node_overrides(&explicit_param_keys);
        merged.apply_defaults();
        merged.apply_variables();
        merged.validate_references()?;
        merged.validate()?;
        Ok(merged)
    }

    fn merge_included(&mut self, other: GraphSpec) {
        self.variables.extend(other.variables);
        self.defaults.backend = other.defaults.backend.or(self.defaults.backend.take());
        self.defaults.device = other.defaults.device.or(self.defaults.device.take());
        self.defaults.precision = other.defaults.precision.or(self.defaults.precision.take());
        self.templates.extend(other.templates);
        self.nodes.extend(other.nodes);
        self.connections.extend(other.connections);
        self.allow_cycles |= other.allow_cycles;
        self.execution = other.execution;
        self.api_version = other.api_version;
        self.kind = other.kind;
    }

    fn apply_templates(&mut self) -> Result<()> {
        for node in &mut self.nodes {
            if let Some(template_name) = node.template.as_ref() {
                let template =
                    self.templates
                        .get(template_name)
                        .ok_or_else(|| Error::Validation {
                            path: format!("nodes[{}].template", node.name),
                            message: format!("unknown template `{template_name}`"),
                        })?;
                node.kind = template.kind.clone();
                node.params = merge_values(template.params.clone(), node.params.clone());
            }
        }
        Ok(())
    }

    fn apply_node_overrides(&mut self, explicit_param_keys: &[BTreeSet<String>]) {
        for (node, explicit_keys) in self.nodes.iter_mut().zip(explicit_param_keys) {
            let Some(descriptor) = find_element(&node.kind) else {
                continue;
            };
            let allowed = |name: &str| descriptor.params.iter().any(|field| field.name == name);
            let values = [
                ("backend", node.backend.as_ref()),
                ("device", node.device.as_ref()),
                ("precision", node.precision.as_ref()),
            ];
            if !values.iter().any(|(_, value)| value.is_some()) {
                continue;
            }
            if node.params.is_null() {
                node.params = Value::Object(Map::new());
            }
            let Value::Object(params) = &mut node.params else {
                continue;
            };
            for (name, value) in values {
                if allowed(name) && !explicit_keys.contains(name) {
                    if let Some(value) = value {
                        params.insert(name.to_string(), Value::String(value.clone()));
                    }
                }
            }
        }
    }

    fn apply_defaults(&mut self) {
        let defaults = self.defaults.clone();
        for node in &mut self.nodes {
            let Some(descriptor) = find_element(&node.kind) else {
                continue;
            };
            let allowed = |name: &str| descriptor.params.iter().any(|field| field.name == name);
            let mut values = [
                ("backend", defaults.backend.as_deref()),
                ("precision", defaults.precision.as_deref()),
            ];
            let has_named_device = matches!(defaults.device, Some(DeviceDefault::Named(_)));
            if !values.iter().any(|(_, value)| value.is_some()) && !has_named_device {
                continue;
            }
            if node.params.is_null() {
                node.params = Value::Object(Map::new());
            }
            let Value::Object(params) = &mut node.params else {
                continue;
            };
            for (name, value) in &mut values {
                if let Some(value) = value.take() {
                    if allowed(name) && !params.contains_key(*name) {
                        params.insert((*name).to_string(), Value::String(value.to_string()));
                    }
                }
            }
            if allowed("device") && !params.contains_key("device") {
                if let Some(DeviceDefault::Named(value)) = defaults.device.as_ref() {
                    params.insert("device".to_string(), Value::String(value.clone()));
                }
            }
        }
    }

    fn apply_variables(&mut self) {
        for node in &mut self.nodes {
            node.params = substitute_variables(node.params.clone(), &self.variables);
        }
        for template in self.templates.values_mut() {
            template.params = substitute_variables(template.params.clone(), &self.variables);
        }
        self.connections = self
            .connections
            .iter()
            .map(|connection| substitute_string(connection, &self.variables))
            .collect();
    }

    fn validate_references(&self) -> Result<()> {
        for node in &self.nodes {
            if let Some(placeholder) = find_unresolved_placeholder(&node.params) {
                return Err(Error::Validation {
                    path: format!("nodes[{}].params", node.name),
                    message: format!("unresolved variable placeholder `{placeholder}`"),
                });
            }
        }
        for (name, template) in &self.templates {
            if let Some(placeholder) = find_unresolved_placeholder(&template.params) {
                return Err(Error::Validation {
                    path: format!("templates[{name}].params"),
                    message: format!("unresolved variable placeholder `{placeholder}`"),
                });
            }
        }
        for (index, connection) in self.connections.iter().enumerate() {
            if let Some(placeholder) = find_unresolved_placeholder_in_string(connection) {
                return Err(Error::Validation {
                    path: format!("connections[{index}]"),
                    message: format!("unresolved variable placeholder `{placeholder}`"),
                });
            }
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.execution.queue_capacity == 0 {
            return Err(Error::Validation {
                path: "execution.queue_capacity".to_string(),
                message: "queue_capacity must be at least 1".to_string(),
            });
        }
        match (self.execution.parallel, self.execution.workers) {
            (_, Some(0)) => {
                return Err(Error::Validation {
                    path: "execution.workers".to_string(),
                    message: "workers must be at least 1".to_string(),
                });
            }
            (ParallelType::Sequential | ParallelType::Pipeline, Some(_)) => {
                return Err(Error::Validation {
                    path: "execution.workers".to_string(),
                    message: "workers is only supported with task parallelism".to_string(),
                });
            }
            _ => {}
        }
        let mut seen = BTreeSet::new();
        for node in &self.nodes {
            if !seen.insert(&node.name) {
                return Err(Error::DuplicateNode(node.name.clone()));
            }
            if node.threads == Some(0) {
                return Err(Error::Validation {
                    path: format!("nodes[{}].threads", node.name),
                    message: "threads must be >= 1".to_string(),
                });
            }
            if self.execution.parallel != ParallelType::Pipeline
                && node.threads.is_some_and(|threads| threads > 1)
            {
                return Err(Error::Validation {
                    path: format!("nodes[{}].threads", node.name),
                    message: "threads > 1 requires Pipeline execution".to_string(),
                });
            }
            element_ports(&node.kind)?;
            validate_element(node).map_err(|err| Error::Validation {
                path: format!("nodes[{}].params", node.name),
                message: err.to_string(),
            })?;
        }

        let mut node_kinds = BTreeMap::new();
        for node in &self.nodes {
            node_kinds.insert(node.name.as_str(), node.kind.as_str());
        }

        let mut edges = Vec::with_capacity(self.connections.len());
        let mut seen_edges = BTreeSet::new();
        let mut connected_inputs = BTreeSet::new();
        for (connection_index, connection) in self.connections.iter().enumerate() {
            let parsed = ConnectionSpec::parse(connection)?;
            let from_kind =
                node_kinds
                    .get(parsed.from_node.as_str())
                    .ok_or_else(|| Error::Validation {
                        path: format!("connections[{connection}]"),
                        message: format!("unknown source node {}", parsed.from_node),
                    })?;
            if self
                .nodes
                .iter()
                .any(|node| node.name == parsed.from_node && node.sink)
            {
                return Err(Error::Validation {
                    path: format!("connections[{connection_index}]"),
                    message: format!(
                        "sink node {} cannot have outgoing connection {}",
                        parsed.from_node, connection
                    ),
                });
            }
            let to_kind =
                node_kinds
                    .get(parsed.to_node.as_str())
                    .ok_or_else(|| Error::Validation {
                        path: format!("connections[{connection}]"),
                        message: format!("unknown destination node {}", parsed.to_node),
                    })?;
            let (_, out_ports) = element_ports(from_kind)?;
            let (in_ports, _) = element_ports(to_kind)?;
            let out_schema =
                find_port(out_ports, &parsed.from_port).ok_or_else(|| Error::UnknownPort {
                    node: parsed.from_node.clone(),
                    port: parsed.from_port.clone(),
                })?;
            let in_schema =
                find_port(in_ports, &parsed.to_port).ok_or_else(|| Error::UnknownPort {
                    node: parsed.to_node.clone(),
                    port: parsed.to_port.clone(),
                })?;
            if let (Some(out_dtype), Some(in_dtype)) = (out_schema.dtype, in_schema.dtype) {
                if out_dtype != in_dtype {
                    return Err(Error::PortTypeMismatch {
                        from_node: parsed.from_node,
                        from_port: parsed.from_port,
                        to_node: parsed.to_node,
                        to_port: parsed.to_port,
                    });
                }
            }

            let edge = (
                parsed.from_node.clone(),
                parsed.from_port.clone(),
                parsed.to_node.clone(),
                parsed.to_port.clone(),
            );
            let path = format!("connections[{connection_index}]");
            if !seen_edges.insert(edge) {
                return Err(Error::Validation {
                    path,
                    message: format!("duplicate connection: {parsed}"),
                });
            }
            let input = (parsed.to_node.clone(), parsed.to_port.clone());
            if !connected_inputs.insert(input) {
                return Err(Error::Validation {
                    path,
                    message: format!(
                        "input port {}.{} already has an incoming connection",
                        parsed.to_node, parsed.to_port
                    ),
                });
            }
            edges.push((parsed.from_node, parsed.to_node));
        }

        for node in &self.nodes {
            let (in_ports, _) = element_ports(&node.kind)?;
            for port in in_ports.iter().filter(|port| port.required) {
                if !connected_inputs.contains(&(node.name.clone(), port.name.to_string())) {
                    return Err(Error::Validation {
                        path: format!("nodes[{}].ports[{}]", node.name, port.name),
                        message: format!(
                            "required input port {}.{} has no incoming connection",
                            node.name, port.name
                        ),
                    });
                }
            }
        }

        if !self.allow_cycles && has_cycle(&self.nodes, &edges) {
            return Err(Error::CycleDetected);
        }
        Ok(())
    }
}

fn find_port<'a>(ports: &'a [PortSchema], name: &str) -> Option<&'a PortSchema> {
    ports.iter().find(|port| port.name == name)
}

fn has_cycle(nodes: &[NodeSpec], edges: &[(String, String)]) -> bool {
    let mut adjacency: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for node in nodes {
        adjacency.entry(&node.name).or_default();
    }
    for (from, to) in edges {
        adjacency.entry(from).or_default().push(to);
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Color {
        White,
        Gray,
        Black,
    }

    fn dfs<'a>(
        node: &'a str,
        adjacency: &BTreeMap<&'a str, Vec<&'a str>>,
        colors: &mut BTreeMap<&'a str, Color>,
    ) -> bool {
        colors.insert(node, Color::Gray);
        if let Some(neighbors) = adjacency.get(node) {
            for &neighbor in neighbors {
                match colors.get(neighbor).copied().unwrap_or(Color::White) {
                    Color::Gray => return true,
                    Color::White => {
                        if dfs(neighbor, adjacency, colors) {
                            return true;
                        }
                    }
                    Color::Black => {}
                }
            }
        }
        colors.insert(node, Color::Black);
        false
    }

    let mut colors: BTreeMap<&str, Color> = BTreeMap::new();
    for node in adjacency.keys().copied() {
        if colors.get(node).copied().unwrap_or(Color::White) == Color::White
            && dfs(node, &adjacency, &mut colors)
        {
            return true;
        }
    }
    false
}

fn merge_values(base: Value, overlay: Value) -> Value {
    match (base, overlay) {
        (Value::Object(mut base_map), Value::Object(overlay_map)) => {
            for (key, value) in overlay_map {
                let merged = match base_map.remove(&key) {
                    Some(existing) => merge_values(existing, value),
                    None => value,
                };
                base_map.insert(key, merged);
            }
            Value::Object(base_map)
        }
        (_, overlay) => overlay,
    }
}

fn substitute_variables(value: Value, vars: &BTreeMap<String, Value>) -> Value {
    match value {
        Value::String(string) => {
            if let Some(replacement) =
                vars.get(string.trim().trim_start_matches("${").trim_end_matches('}'))
            {
                if string.starts_with("${")
                    && string.ends_with('}')
                    && string.matches("${").count() == 1
                {
                    return replacement.clone();
                }
            }
            Value::String(substitute_string(&string, vars))
        }
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(|value| substitute_variables(value, vars))
                .collect(),
        ),
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, value)| (key, substitute_variables(value, vars)))
                .collect::<Map<_, _>>(),
        ),
        other => other,
    }
}

fn substitute_string(value: &str, vars: &BTreeMap<String, Value>) -> String {
    let mut out = value.to_string();
    for (key, replacement) in vars {
        let needle = format!("${{{key}}}");
        let replacement = match replacement {
            Value::String(string) => string.clone(),
            other => other.to_string(),
        };
        out = out.replace(&needle, &replacement);
    }
    out
}

fn find_unresolved_placeholder(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => find_unresolved_placeholder_in_string(value),
        Value::Array(values) => values.iter().find_map(find_unresolved_placeholder),
        Value::Object(values) => values.values().find_map(find_unresolved_placeholder),
        _ => None,
    }
}

fn find_unresolved_placeholder_in_string(value: &str) -> Option<String> {
    let start = value.find("${")?;
    let remainder = &value[start + 2..];
    let end = remainder
        .find('}')
        .map_or(value.len(), |end| start + 3 + end);
    Some(value[start..end].to_string())
}

#[derive(Clone, Debug, Default)]
pub struct GraphSpecBuilder {
    spec: GraphSpec,
}

impl GraphSpecBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn api_version(mut self, api_version: impl Into<String>) -> Self {
        self.spec.api_version = api_version.into();
        self
    }

    pub fn allow_cycles(mut self, allow_cycles: bool) -> Self {
        self.spec.allow_cycles = allow_cycles;
        self
    }

    pub fn execution(mut self, execution: ExecutionSpec) -> Self {
        self.spec.execution = execution;
        self
    }

    pub fn defaults(mut self, defaults: DefaultsSpec) -> Self {
        self.spec.defaults = defaults;
        self
    }

    pub fn variable(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.spec.variables.insert(key.into(), value.into());
        self
    }

    pub fn add_template(mut self, name: impl Into<String>, template: NodeTemplate) -> Self {
        self.spec.templates.insert(name.into(), template);
        self
    }

    pub fn add_node(mut self, node: NodeSpec) -> Self {
        self.spec.nodes.push(node);
        self
    }

    pub fn connect(mut self, connection: impl Into<String>) -> Self {
        self.spec.connections.push(connection.into());
        self
    }

    pub fn build(self) -> Result<GraphSpec> {
        self.spec.normalize_with_base_dir(None)
    }
}
