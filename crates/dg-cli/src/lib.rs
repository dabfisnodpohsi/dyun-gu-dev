use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use dg_graph::{Graph, GraphReport, GraphSpec};
use tracing_subscriber::EnvFilter;

use dg_elements as _;

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
    },
    Validate {
        #[arg(long)]
        config: PathBuf,
    },
    ListElements,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum OutputFormat {
    Json,
    Text,
}

pub fn run(cli: Cli) -> Result<()> {
    init_logging(cli.verbose);
    match cli.command {
        Command::Run { config, format } => run_graph(&config, format),
        Command::Validate { config } => validate_graph(&config),
        Command::ListElements => list_elements(),
    }
}

pub fn run_graph(path: &Path, format: OutputFormat) -> Result<()> {
    let spec = load_spec(path)?;
    let graph = Graph::new(spec).context("build graph")?;
    let report = graph.run().context("run graph")?;
    print_report(&report, format)
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

    use super::{list_elements, run_graph, validate_graph, OutputFormat};

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
    kind: mock_inference
    params:
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
        fs::remove_file(path).expect("remove config");
    }

    #[test]
    fn documented_multi_algorithm_example_runs() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/mock-multi-algorithm.yaml");
        validate_graph(&path).expect("validate documented example");
        run_graph(&path, OutputFormat::Json).expect("run documented example");
    }
}
