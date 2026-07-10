use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use dg_graph::{Graph, GraphDiff, GraphReport, GraphSpec};
use tracing_subscriber::EnvFilter;

use dg_elements as _;
#[cfg(feature = "media")]
use dg_media as _;
#[cfg(feature = "openvino")]
use dg_openvino as _;
#[cfg(feature = "rknn")]
use dg_rknn as _;
#[cfg(feature = "sophon")]
use dg_sophon as _;
#[cfg(feature = "stream")]
use dg_stream as _;
#[cfg(feature = "tensorrt")]
use dg_tensorrt as _;

#[derive(Debug, Parser)]
#[command(
    name = "dg",
    version,
    about = "Run and inspect dg graph specifications"
)]
pub struct Cli {
    #[arg(long, global = true, short = 'v', action = clap::ArgAction::Count)]
    pub verbose: u8,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Run {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
        #[arg(long)]
        watch: bool,
    },
    Validate {
        #[arg(long)]
        config: PathBuf,
    },
    ListElements,
    Schema {
        #[arg(long)]
        kind: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum OutputFormat {
    Json,
    Text,
}

pub fn run(cli: Cli) -> Result<()> {
    init_logging(cli.verbose);
    match cli.command {
        Command::Run {
            config,
            format,
            watch,
        } => run_graph_with_watch(&config, format, watch),
        Command::Validate { config } => validate_graph(&config),
        Command::ListElements => list_elements(),
        Command::Schema { kind } => schema(kind.as_deref()),
    }
}

pub fn run_graph(path: &Path, format: OutputFormat) -> Result<()> {
    run_graph_with_watch(path, format, false)
}

fn run_graph_with_watch(path: &Path, format: OutputFormat, watch: bool) -> Result<()> {
    let spec = load_spec(path)?;
    let graph = Graph::new(spec).context("build graph")?;
    let report = graph.run().context("run graph")?;
    print_report(&report, format)?;
    if watch {
        let _watch_handle = dg_graph::watch(path, move |result| match result {
            Ok((_, diff)) if !diff.is_empty() => match render_diff(&diff, format) {
                Ok(output) => println!("{output}"),
                Err(error) => println!("failed to render graph reload diff: {error}"),
            },
            Ok(_) => {}
            Err(error) => println!("{}", render_reload_rejected(&error.to_string())),
        })?;
        loop {
            std::thread::park();
        }
    }
    Ok(())
}

pub fn validate_graph(path: &Path) -> Result<()> {
    let _ = load_spec(path)?;
    println!("valid: {}", path.display());
    Ok(())
}

pub fn list_elements() -> Result<()> {
    let mut kinds = dg_graph::registered_elements()
        .into_iter()
        .map(|descriptor| descriptor.kind)
        .collect::<Vec<_>>();
    kinds.sort_unstable();
    kinds.dedup();
    for kind in kinds {
        println!("{kind}");
    }
    Ok(())
}

pub fn schema(kind: Option<&str>) -> Result<()> {
    let value = match kind {
        Some(kind) => dg_graph::element_params_schema(kind)
            .ok_or_else(|| anyhow::anyhow!("unknown element kind: {kind}"))?,
        None => serde_json::to_value(dg_graph::all_element_schemas())?,
    };
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

fn load_spec(path: &Path) -> Result<GraphSpec> {
    GraphSpec::load_from_path(path).with_context(|| format!("load graph config {}", path.display()))
}

fn print_report(report: &GraphReport, format: OutputFormat) -> Result<()> {
    let summary = ReportSummary::from(report);
    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&summary)?),
        OutputFormat::Text => {
            println!("graph run completed");
            println!("sinks: {}", summary.sinks.len());
            for sink in &summary.sinks {
                println!(
                    "  {}: {} tensor(s), {} detection(s), {} classification(s), \
                     {} face(s), {} track(s), {} OCR result(s)",
                    sink.name,
                    sink.tensors,
                    sink.detections,
                    sink.classifications,
                    sink.faces,
                    sink.tracks,
                    sink.ocr
                );
            }
        }
    }
    Ok(())
}

#[derive(Debug, serde::Serialize)]
struct DiffSummary {
    added_nodes: Vec<String>,
    removed_nodes: Vec<String>,
    updated_nodes: Vec<String>,
    added_connections: Vec<String>,
    removed_connections: Vec<String>,
}

fn diff_summary(diff: &GraphDiff) -> DiffSummary {
    DiffSummary {
        added_nodes: diff
            .added_nodes
            .iter()
            .map(|node| node.name.clone())
            .collect(),
        removed_nodes: diff.removed_nodes.clone(),
        updated_nodes: diff
            .updated_nodes
            .iter()
            .map(|node| node.name.clone())
            .collect(),
        added_connections: diff.added_connections.clone(),
        removed_connections: diff.removed_connections.clone(),
    }
}

fn render_diff(diff: &GraphDiff, format: OutputFormat) -> Result<String> {
    let summary = diff_summary(diff);
    match format {
        OutputFormat::Json => Ok(serde_json::to_string_pretty(&summary)?),
        OutputFormat::Text => {
            let mut lines = vec!["graph configuration reloaded".to_string()];
            if !summary.added_nodes.is_empty() {
                lines.push(format!("added nodes: {}", summary.added_nodes.join(", ")));
            }
            if !summary.removed_nodes.is_empty() {
                lines.push(format!(
                    "removed nodes: {}",
                    summary.removed_nodes.join(", ")
                ));
            }
            if !summary.updated_nodes.is_empty() {
                lines.push(format!(
                    "updated nodes: {}",
                    summary.updated_nodes.join(", ")
                ));
            }
            if !summary.added_connections.is_empty() {
                lines.push(format!(
                    "added connections: {}",
                    summary.added_connections.join(", ")
                ));
            }
            if !summary.removed_connections.is_empty() {
                lines.push(format!(
                    "removed connections: {}",
                    summary.removed_connections.join(", ")
                ));
            }
            Ok(lines.join("\n"))
        }
    }
}

fn render_reload_rejected(error: &str) -> String {
    format!("graph configuration reload REJECTED: {error}; previous configuration remains active")
}

#[derive(Debug, serde::Serialize)]
struct ReportSummary {
    sinks: Vec<SinkSummary>,
}

#[derive(Debug, serde::Serialize)]
struct SinkSummary {
    name: String,
    tensors: usize,
    detections: usize,
    classifications: usize,
    faces: usize,
    tracks: usize,
    ocr: usize,
}

impl From<&GraphReport> for ReportSummary {
    fn from(report: &GraphReport) -> Self {
        let mut names = report
            .sinks
            .keys()
            .chain(report.detections.keys())
            .chain(report.classifications.keys())
            .chain(report.faces.keys())
            .chain(report.tracks.keys())
            .chain(report.ocr.keys())
            .cloned()
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        let sinks = names
            .into_iter()
            .map(|name| SinkSummary {
                tensors: report.sinks.get(&name).map_or(0, Vec::len),
                detections: report.detections.get(&name).map_or(0, Vec::len),
                classifications: report.classifications.get(&name).map_or(0, Vec::len),
                faces: report.faces.get(&name).map_or(0, Vec::len),
                tracks: report.tracks.get(&name).map_or(0, Vec::len),
                ocr: report.ocr.get(&name).map_or(0, Vec::len),
                name,
            })
            .collect();
        Self { sinks }
    }
}

fn init_logging(verbose: u8) {
    let default = match verbose {
        0 => "warn",
        1 => "info",
        _ => "debug",
    };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

#[cfg(test)]
mod tests {
    use std::fs;

    use dg_graph::{GraphDiff, NodeSpec};

    #[cfg(feature = "stream")]
    use super::Command;
    use super::{
        list_elements, render_diff, render_reload_rejected, run_graph, schema, validate_graph,
        OutputFormat,
    };

    fn temp_config() -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "dg-cli-{}-{}.yaml",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let content = r#"
apiVersion: dg/v1
kind: Graph
nodes:
  - name: source
    kind: source
    params:
      count: 1
      shape: [1, 2]
  - name: infer
    kind: inference
    params:
      backend: mock
      options:
        shape: [1, 2]
        echo_inputs: true
  - name: sink
    kind: sink
    params: {}
connections:
  - source.out -> infer.in
  - infer.out -> sink.in
"#;
        fs::write(&path, content).expect("write config");
        path
    }

    #[test]
    fn commands_run_validate_and_list_elements() {
        let path = temp_config();
        validate_graph(&path).expect("validate config");
        run_graph(&path, OutputFormat::Json).expect("run config");
        list_elements().expect("list elements");
        #[cfg(feature = "stream")]
        {
            let kinds = dg_graph::registered_elements()
                .into_iter()
                .map(|descriptor| descriptor.kind)
                .collect::<std::collections::BTreeSet<_>>();
            for kind in [
                "media_decode",
                "media_encode",
                "media_resize",
                "media_osd",
                "rtsp_src",
                "httpflv_src",
                "rtmp_sink",
                "webrtc_sink",
            ] {
                assert!(kinds.contains(kind), "missing registered element {kind}");
            }
        }
        fs::remove_file(path).expect("remove config");
    }

    #[test]
    fn documented_multi_algorithm_example_runs() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/mock-multi-algorithm.yaml");
        validate_graph(&path).expect("validate documented example");
        run_graph(&path, OutputFormat::Json).expect("run documented example");
    }

    #[test]
    fn schema_command_exports_all_and_one_element() {
        schema(None).expect("export all element schemas");
        #[cfg(feature = "stream")]
        {
            schema(Some("media_osd")).expect("export media OSD schema");
            let command = Command::Schema {
                kind: Some("media_osd".to_string()),
            };
            assert!(matches!(command, Command::Schema { .. }));
            let schema = dg_graph::element_params_schema("media_osd").expect("media OSD schema");
            assert_eq!(schema["properties"]["boxes"]["type"], "array");
        }
    }

    #[test]
    fn diff_rendering_supports_text_and_json() {
        let diff = GraphDiff {
            added_nodes: vec![NodeSpec {
                name: "added".to_string(),
                kind: "source".to_string(),
                template: None,
                params: serde_json::json!({}),
                ..NodeSpec::default()
            }],
            removed_nodes: vec!["removed".to_string()],
            updated_nodes: vec![NodeSpec {
                name: "updated".to_string(),
                kind: "sink".to_string(),
                template: None,
                params: serde_json::json!({}),
                ..NodeSpec::default()
            }],
            added_connections: vec!["added.out -> updated.in".to_string()],
            removed_connections: vec!["old.out -> removed.in".to_string()],
        };

        let text = render_diff(&diff, OutputFormat::Text).expect("render text diff");
        assert!(text.contains("added nodes: added"));
        assert!(text.contains("removed nodes: removed"));
        assert!(text.contains("updated nodes: updated"));
        assert!(text.contains("added connections: added.out -> updated.in"));
        assert!(text.contains("removed connections: old.out -> removed.in"));

        let json = render_diff(&diff, OutputFormat::Json).expect("render JSON diff");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse JSON diff");
        assert_eq!(value["added_nodes"], serde_json::json!(["added"]));
        assert_eq!(value["removed_nodes"], serde_json::json!(["removed"]));
        assert_eq!(value["updated_nodes"], serde_json::json!(["updated"]));
        assert_eq!(
            value["added_connections"],
            serde_json::json!(["added.out -> updated.in"])
        );
        assert_eq!(
            value["removed_connections"],
            serde_json::json!(["old.out -> removed.in"])
        );
    }

    #[test]
    fn invalid_reload_message_keeps_previous_configuration() {
        let message = render_reload_rejected("invalid node parameters");
        assert!(message.contains("REJECTED"));
        assert!(message.contains("invalid node parameters"));
        assert!(message.contains("previous configuration remains active"));
    }

    #[cfg(feature = "openvino")]
    #[test]
    fn openvino_feature_registers_configuration() {
        let config = dg_runtime::BackendConfig::new(
            Some(std::path::PathBuf::from("model.xml")),
            serde_json::json!({"device": "GPU"}),
        );
        let option = dg_runtime::configure_backend("openvino", config).expect("configure OpenVINO");
        assert_eq!(option.backend, dg_runtime::BackendKind::OpenVINO);
        assert_eq!(
            option
                .backend_options
                .as_openvino()
                .expect("OpenVINO options")
                .device,
            "GPU"
        );
    }

    #[cfg(feature = "openvino")]
    #[test]
    fn validate_rejects_openvino_capability_mismatch_without_initializing_model() {
        let path = std::env::temp_dir().join(format!(
            "dg-cli-openvino-preflight-{}.yaml",
            std::process::id()
        ));
        let content = r#"
apiVersion: dg/v1
kind: Graph
nodes:
  - name: infer
    kind: inference
    params:
      backend: openvino
      model: missing.xml
      device: cuda_gpu
"#;
        fs::write(&path, content).expect("write config");
        let err = validate_graph(&path).expect_err("device should fail preflight");
        fs::remove_file(path).expect("remove config");
        assert!(format!("{err:#}").contains("unsupported device: CudaGpu"));
    }
}
