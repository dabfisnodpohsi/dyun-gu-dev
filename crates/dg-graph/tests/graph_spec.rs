use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use dg_graph::{
    watch, ConnectionSpec, Graph, GraphFormat, GraphSpec, GraphSpecBuilder, NodeSpec, NodeTemplate,
};
use proptest::prelude::*;
use serde_json::json;

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be monotonic")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{nanos}-{}", std::process::id()))
}

fn sample_spec() -> GraphSpec {
    GraphSpecBuilder::new()
        .variable("start", json!(7.0))
        .add_template(
            "mock_base",
            NodeTemplate {
                kind: "mock_inference".to_string(),
                template: None,
                params: json!({
                    "shape": [1, 4],
                    "echo_inputs": true
                }),
            },
        )
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            template: None,
            params: json!({
                "count": 2,
                "shape": [1, 4],
                "start": "${start}"
            }),
        })
        .add_node(NodeSpec {
            name: "infer".to_string(),
            kind: "mock_inference".to_string(),
            template: Some("mock_base".to_string()),
            params: json!({
                "fill_value": 0
            }),
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
        })
        .connect("source.out -> infer.in")
        .connect("infer.out -> sink.in")
        .build()
        .expect("build sample spec")
}

fn variant_spec(
    source_count: usize,
    source_start: f64,
    echo_inputs: bool,
    with_extra_pipeline: bool,
) -> GraphSpec {
    let mut builder = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            template: None,
            params: json!({
                "count": source_count,
                "shape": [1, 4],
                "start": source_start
            }),
        })
        .add_node(NodeSpec {
            name: "infer".to_string(),
            kind: "mock_inference".to_string(),
            template: None,
            params: json!({
                "shape": [1, 4],
                "echo_inputs": echo_inputs
            }),
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
        })
        .connect("source.out -> infer.in")
        .connect("infer.out -> sink.in");
    if with_extra_pipeline {
        builder = builder
            .add_node(NodeSpec {
                name: "extra_source".to_string(),
                kind: "source".to_string(),
                template: None,
                params: json!({"count": 0, "shape": [1, 4]}),
            })
            .add_node(NodeSpec {
                name: "extra_sink".to_string(),
                kind: "sink".to_string(),
                template: None,
                params: json!({}),
            })
            .connect("extra_source.out -> extra_sink.in");
    }
    builder.build().expect("build variant graph spec")
}

fn semantic_nodes(spec: &GraphSpec) -> BTreeMap<String, NodeSpec> {
    spec.nodes
        .iter()
        .map(|node| (node.name.clone(), node.clone()))
        .collect()
}

fn semantic_connections(spec: &GraphSpec) -> BTreeSet<String> {
    spec.connections.iter().cloned().collect()
}

proptest! {
    #[test]
    fn graph_diff_apply_preserves_spec_semantics(
        source_count_a in 0_usize..4,
        source_count_b in 0_usize..4,
        source_start_a in 0_i32..8,
        source_start_b in 0_i32..8,
        echo_a in any::<bool>(),
        echo_b in any::<bool>(),
        extra_a in any::<bool>(),
        extra_b in any::<bool>(),
    ) {
        let a = variant_spec(
            source_count_a,
            f64::from(source_start_a),
            echo_a,
            extra_a,
        );
        let b = variant_spec(
            source_count_b,
            f64::from(source_start_b),
            echo_b,
            extra_b,
        );
        let diff = Graph::diff(&a, &b);
        let mut graph = Graph::new(a.clone()).expect("build source graph");
        diff.clone().apply(&mut graph).expect("apply graph diff");

        prop_assert_eq!(semantic_nodes(graph.spec()), semantic_nodes(&b));
        prop_assert_eq!(semantic_connections(graph.spec()), semantic_connections(&b));

        let mut reloaded = Graph::new(a.clone()).expect("build source graph");
        let reloaded_diff = reloaded.reload(b.clone()).expect("reload graph");
        prop_assert_eq!(reloaded_diff, diff);
    }
}

#[test]
fn graph_spec_round_trips_across_yaml_json_and_toml() {
    let spec = sample_spec();
    for format in [GraphFormat::Yaml, GraphFormat::Json, GraphFormat::Toml] {
        let encoded = spec
            .to_string_with_format(format)
            .expect("serialize graph spec");
        let decoded = GraphSpec::from_str_with_format(&encoded, format).expect("parse graph spec");
        assert_eq!(decoded, spec);
    }
}

#[test]
fn graph_spec_validation_rejects_duplicate_names_and_cycles() {
    let duplicate = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "dup".to_string(),
            kind: "source".to_string(),
            template: None,
            params: json!({"count": 1, "shape": [1, 4]}),
        })
        .add_node(NodeSpec {
            name: "dup".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
        })
        .build();
    assert!(duplicate.is_err());

    let cycle = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "a".to_string(),
            kind: "mock_inference".to_string(),
            template: None,
            params: json!({"shape": [1, 4]}),
        })
        .add_node(NodeSpec {
            name: "b".to_string(),
            kind: "mock_inference".to_string(),
            template: None,
            params: json!({"shape": [1, 4]}),
        })
        .connect("a.out -> b.in")
        .connect("b.out -> a.in")
        .build();
    assert!(cycle.is_err());
}

#[test]
fn graph_spec_rejects_hanging_references() {
    let spec = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            template: None,
            params: json!({"count": 1, "shape": [1, 4]}),
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
        })
        .connect("source.out -> missing.in")
        .build();
    assert!(spec.is_err());
}

#[test]
fn graph_spec_connection_parse_round_trips() {
    let parsed = ConnectionSpec::parse("decode.out -> infer.in").expect("parse connection");
    assert_eq!(parsed.to_string(), "decode.out -> infer.in");
}

proptest! {
    #[test]
    fn graph_spec_connection_round_trip_property(
        from_node in "[a-z][a-z0-9_]{0,6}",
        from_port in "[a-z][a-z0-9_]{0,6}",
        to_node in "[a-z][a-z0-9_]{0,6}",
        to_port in "[a-z][a-z0-9_]{0,6}"
    ) {
        let spec = format!("{from_node}.{from_port} -> {to_node}.{to_port}");
        let parsed = ConnectionSpec::parse(&spec).expect("parse generated connection");
        prop_assert_eq!(parsed.to_string(), spec);
    }
}

#[test]
fn graph_spec_loads_includes_and_templates_from_yaml() {
    let root = unique_temp_dir("dg-graph-spec");
    fs::create_dir_all(&root).expect("create temp dir");
    fs::write(
        root.join("common.yaml"),
        r#"
apiVersion: dg/v1
kind: Graph
variables:
  start: 5
templates:
  mock_base:
    kind: mock_inference
    params:
      shape: [1, 4]
      echo_inputs: true
"#,
    )
    .expect("write include");
    fs::write(
        root.join("graph.yaml"),
        r#"
apiVersion: dg/v1
kind: Graph
includes: ["common.yaml"]
nodes:
  - name: source
    kind: source
    params:
      count: 1
      shape: [1, 4]
      start: "${start}"
  - name: infer
    kind: mock_inference
    template: mock_base
    params:
      fill_value: 0
  - name: sink
    kind: sink
    params: {}
connections:
  - "source.out -> infer.in"
  - "infer.out -> sink.in"
"#,
    )
    .expect("write graph");

    let spec = GraphSpec::load_from_path(root.join("graph.yaml")).expect("load graph spec");
    assert_eq!(spec.variables.get("start"), Some(&json!(5)));
    assert_eq!(spec.nodes.len(), 3);
    assert_eq!(spec.nodes[0].params["start"], json!(5));
    assert_eq!(spec.nodes[1].kind, "mock_inference");
    assert_eq!(spec.nodes[1].params["shape"], json!([1, 4]));
}

#[test]
fn graph_watch_reports_valid_reload_and_diff() {
    let root = unique_temp_dir("dg-graph-watch");
    fs::create_dir_all(&root).expect("create temp dir");
    let path = root.join("graph.yaml");
    let first = variant_spec(0, 1.0, true, false);
    let second = variant_spec(1, 2.0, false, true);
    fs::write(
        &path,
        first
            .to_string_with_format(GraphFormat::Yaml)
            .expect("serialize initial graph"),
    )
    .expect("write initial graph");

    let (sender, receiver) = mpsc::channel();
    let handle = watch(&path, move |event| {
        sender.send(event).expect("send watch event");
    })
    .expect("start graph watch");
    std::thread::sleep(Duration::from_millis(75));
    fs::write(
        &path,
        second
            .to_string_with_format(GraphFormat::Yaml)
            .expect("serialize updated graph"),
    )
    .expect("write updated graph");

    let event = receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("receive graph watch event")
        .expect("valid graph watch event");
    assert_eq!(event.0, second);
    assert_eq!(event.1, Graph::diff(&first, &second));
    handle.stop();
    fs::remove_dir_all(root).expect("remove watch temp dir");
}

#[test]
fn graph_diff_is_empty_for_identical_specs() {
    let spec = sample_spec();
    let diff = Graph::diff(&spec, &spec);
    assert!(diff.is_empty());
}
