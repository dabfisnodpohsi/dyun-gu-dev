use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use dg_graph::{
    watch, ConnectionSpec, DefaultsSpec, DeviceDefault, ExecutionSpec, Graph, GraphFormat,
    GraphSpec, GraphSpecBuilder, NodeSpec, NodeTemplate, ParallelType,
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
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "infer".to_string(),
            kind: "mock_inference".to_string(),
            template: Some("mock_base".to_string()),
            params: json!({
                "fill_value": 0
            }),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
            ..NodeSpec::default()
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
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "infer".to_string(),
            kind: "mock_inference".to_string(),
            template: None,
            params: json!({
                "shape": [1, 4],
                "echo_inputs": echo_inputs
            }),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
            ..NodeSpec::default()
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
                ..NodeSpec::default()
            })
            .add_node(NodeSpec {
                name: "extra_sink".to_string(),
                kind: "sink".to_string(),
                template: None,
                params: json!({}),
                ..NodeSpec::default()
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

fn inference_graph() -> GraphSpecBuilder {
    GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            template: None,
            params: json!({"count": 1, "shape": [1, 4]}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "infer".to_string(),
            kind: "inference".to_string(),
            template: None,
            params: json!({}),
            ..NodeSpec::default()
        })
        .connect("source.out -> infer.in")
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
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "dup".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
            ..NodeSpec::default()
        })
        .build();
    assert!(duplicate.is_err());

    let cycle = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "a".to_string(),
            kind: "mock_inference".to_string(),
            template: None,
            params: json!({"shape": [1, 4]}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "b".to_string(),
            kind: "mock_inference".to_string(),
            template: None,
            params: json!({"shape": [1, 4]}),
            ..NodeSpec::default()
        })
        .connect("a.out -> b.in")
        .connect("b.out -> a.in")
        .build();
    assert!(cycle.is_err());
}

#[test]
fn cfg09_rejects_unknown_template_references() {
    let error = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "infer".to_string(),
            kind: "mock_inference".to_string(),
            template: Some("missing".to_string()),
            params: json!({"shape": [1, 4]}),
            ..NodeSpec::default()
        })
        .build()
        .expect_err("unknown templates should be rejected");
    assert!(error.to_string().contains("nodes[infer].template"));
    assert!(error.to_string().contains("missing"));
}

#[test]
fn cfg09_rejects_unresolved_variables_in_params_and_connections() {
    let param_error = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            params: json!({
                "count": 1,
                "shape": [1, 4],
                "start": "${undefined}"
            }),
            ..NodeSpec::default()
        })
        .build()
        .expect_err("unresolved parameter variables should be rejected");
    assert!(param_error.to_string().contains("nodes[source].params"));
    assert!(param_error.to_string().contains("${undefined}"));

    let connection_error = GraphSpec {
        nodes: vec![
            NodeSpec {
                name: "source".to_string(),
                kind: "source".to_string(),
                params: json!({"count": 1, "shape": [1, 4]}),
                ..NodeSpec::default()
            },
            NodeSpec {
                name: "sink".to_string(),
                kind: "sink".to_string(),
                params: json!({}),
                ..NodeSpec::default()
            },
        ],
        connections: vec!["source.out -> ${undefined}.in".to_string()],
        ..GraphSpec::default()
    }
    .normalize_with_base_dir(None)
    .expect_err("unresolved connection variables should be rejected");
    assert!(connection_error.to_string().contains("connections[0]"));
    assert!(connection_error.to_string().contains("${undefined}"));
}

#[test]
fn cfg09_rejects_includes_without_a_base_directory() {
    let error = GraphSpec {
        includes: vec!["common.yaml".to_string()],
        ..GraphSpec::default()
    }
    .normalize_with_base_dir(None)
    .expect_err("includes need a base directory");
    assert!(error.to_string().contains("includes"));
    assert!(error.to_string().contains("base directory"));
}

#[test]
fn cfg09_rejects_include_cycles() {
    let root = unique_temp_dir("dg-graph-include-cycle");
    fs::create_dir_all(&root).expect("create temp dir");
    fs::write(
        root.join("a.yaml"),
        r#"
apiVersion: dg/v1
kind: Graph
includes: ["b.yaml"]
"#,
    )
    .expect("write first cyclic include");
    fs::write(
        root.join("b.yaml"),
        r#"
apiVersion: dg/v1
kind: Graph
includes: ["a.yaml"]
"#,
    )
    .expect("write second cyclic include");

    let error = GraphSpec::load_from_path(root.join("a.yaml"))
        .expect_err("include cycles should be rejected");
    assert!(error.to_string().contains("include cycle detected"));
    fs::remove_dir_all(root).expect("remove temp dir");
}

#[test]
fn cfg08_validates_threads_and_sink_semantics() {
    let threads_zero = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "infer".to_string(),
            kind: "mock_inference".to_string(),
            threads: Some(0),
            params: json!({"shape": [1, 4]}),
            ..NodeSpec::default()
        })
        .build()
        .expect_err("zero threads should be rejected");
    assert!(threads_zero.to_string().contains("threads must be >= 1"));

    let sink_with_output = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            params: json!({"count": 1, "shape": [1, 4]}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "terminal".to_string(),
            kind: "mock_inference".to_string(),
            sink: true,
            params: json!({"shape": [1, 4]}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            params: json!({}),
            ..NodeSpec::default()
        })
        .connect("source.out -> terminal.in")
        .connect("terminal.out -> sink.in")
        .build()
        .expect_err("terminal nodes should not have outgoing edges");
    assert!(sink_with_output
        .to_string()
        .contains("cannot have outgoing connection"));

    for parallel in [ParallelType::Sequential, ParallelType::Task] {
        let error = GraphSpecBuilder::new()
            .execution(ExecutionSpec {
                parallel,
                ..ExecutionSpec::default()
            })
            .add_node(NodeSpec {
                name: "infer".to_string(),
                kind: "mock_inference".to_string(),
                threads: Some(2),
                params: json!({"shape": [1, 4]}),
                ..NodeSpec::default()
            })
            .build()
            .expect_err("multi-instancing must be Pipeline-only");
        assert!(error
            .to_string()
            .contains("threads > 1 requires Pipeline execution"));
    }

    GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            params: json!({"count": 1, "shape": [1, 4]}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "infer".to_string(),
            kind: "mock_inference".to_string(),
            threads: Some(2),
            params: json!({"shape": [1, 4]}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "terminal".to_string(),
            kind: "sink".to_string(),
            sink: true,
            params: json!({}),
            ..NodeSpec::default()
        })
        .connect("source.out -> infer.in")
        .connect("infer.out -> terminal.in")
        .build()
        .expect("Pipeline multi-instancing with a terminal sink should validate");
}

#[test]
fn graph_defaults_fill_inference_parameters() {
    let spec = inference_graph()
        .defaults(DefaultsSpec {
            backend: Some("mock".to_string()),
            device: Some(DeviceDefault::Named("cpu".to_string())),
            precision: Some("f32".to_string()),
        })
        .build()
        .expect("defaults should complete inference parameters");
    let params = &spec.nodes[1].params;
    assert_eq!(params["backend"], "mock");
    assert_eq!(params["device"], "cpu");
    assert_eq!(params["precision"], "f32");
}

#[test]
fn graph_defaults_do_not_override_node_parameters() {
    let spec = GraphSpec {
        defaults: DefaultsSpec {
            backend: Some("mock".to_string()),
            device: Some(DeviceDefault::Named("cpu".to_string())),
            precision: Some("f32".to_string()),
        },
        nodes: vec![
            NodeSpec {
                name: "source".to_string(),
                kind: "source".to_string(),
                template: None,
                params: json!({"count": 1, "shape": [1, 4]}),
                ..NodeSpec::default()
            },
            NodeSpec {
                name: "infer".to_string(),
                kind: "inference".to_string(),
                template: None,
                params: json!({
                    "backend": "mock",
                    "device": "cpu",
                    "precision": "f16"
                }),
                ..NodeSpec::default()
            },
        ],
        connections: vec!["source.out -> infer.in".to_string()],
        ..GraphSpec::default()
    }
    .normalize_with_base_dir(None)
    .expect("explicit parameters should remain valid");
    assert_eq!(spec.nodes[1].params["precision"], "f16");
}

#[test]
fn graph_defaults_do_not_override_template_parameters() {
    let spec = GraphSpec {
        defaults: DefaultsSpec {
            backend: Some("mock".to_string()),
            device: Some(DeviceDefault::Named("cpu".to_string())),
            precision: Some("f32".to_string()),
        },
        templates: BTreeMap::from([(
            "inference_defaults".to_string(),
            NodeTemplate {
                kind: "inference".to_string(),
                template: None,
                params: json!({"precision": "f16"}),
            },
        )]),
        nodes: vec![
            NodeSpec {
                name: "source".to_string(),
                kind: "source".to_string(),
                template: None,
                params: json!({"count": 1, "shape": [1, 4]}),
                ..NodeSpec::default()
            },
            NodeSpec {
                name: "infer".to_string(),
                kind: "inference".to_string(),
                template: Some("inference_defaults".to_string()),
                params: json!({}),
                ..NodeSpec::default()
            },
        ],
        connections: vec!["source.out -> infer.in".to_string()],
        ..GraphSpec::default()
    }
    .normalize_with_base_dir(None)
    .expect("template parameters should remain valid");
    assert_eq!(spec.nodes[1].params["precision"], "f16");
}

#[test]
fn node_fields_override_template_parameters() {
    let spec = GraphSpec {
        templates: BTreeMap::from([(
            "inference_defaults".to_string(),
            NodeTemplate {
                kind: "inference".to_string(),
                template: None,
                params: json!({"precision": "f16"}),
            },
        )]),
        nodes: vec![
            NodeSpec {
                name: "source".to_string(),
                kind: "source".to_string(),
                params: json!({"count": 1, "shape": [1, 4]}),
                ..NodeSpec::default()
            },
            NodeSpec {
                name: "infer".to_string(),
                kind: "inference".to_string(),
                template: Some("inference_defaults".to_string()),
                precision: Some("f32".to_string()),
                params: json!({}),
                ..NodeSpec::default()
            },
        ],
        connections: vec!["source.out -> infer.in".to_string()],
        defaults: DefaultsSpec {
            backend: Some("mock".to_string()),
            device: Some(DeviceDefault::Named("cpu".to_string())),
            precision: Some("f16".to_string()),
        },
        ..GraphSpec::default()
    }
    .normalize_with_base_dir(None)
    .expect("node-level fields should override template parameters");
    assert_eq!(spec.nodes[1].params["precision"], "f32");
}

#[test]
fn graph_defaults_from_includes_have_lower_precedence() {
    let root = unique_temp_dir("dg-graph-defaults");
    fs::create_dir_all(&root).expect("create temp dir");
    fs::write(
        root.join("common.yaml"),
        r#"
apiVersion: dg/v1
kind: Graph
defaults:
  backend: mock
  precision: f16
"#,
    )
    .expect("write include");
    fs::write(
        root.join("graph.yaml"),
        r#"
apiVersion: dg/v1
kind: Graph
includes: ["common.yaml"]
defaults:
  precision: f32
  device: cpu
nodes:
  - name: source
    kind: source
    params:
      count: 1
      shape: [1, 4]
  - name: infer
    kind: inference
    params: {}
connections:
  - source.out -> infer.in
"#,
    )
    .expect("write graph");

    let spec = GraphSpec::load_from_path(root.join("graph.yaml")).expect("load graph");
    let params = &spec.nodes[1].params;
    assert_eq!(params["backend"], "mock");
    assert_eq!(params["precision"], "f32");
    assert_eq!(params["device"], "cpu");
    fs::remove_dir_all(root).expect("remove temp dir");
}

#[test]
fn graph_defaults_substitute_variables() {
    let spec = inference_graph()
        .variable("default_backend", "mock")
        .defaults(DefaultsSpec {
            backend: Some("${default_backend}".to_string()),
            device: Some(DeviceDefault::Named("cpu".to_string())),
            precision: Some("f32".to_string()),
        })
        .build()
        .expect("variable-backed defaults should validate");
    assert_eq!(spec.nodes[1].params["backend"], "mock");
}

#[test]
fn graph_defaults_reject_unknown_fields() {
    let err = GraphSpec::from_str_with_format(
        r#"
apiVersion: dg/v1
kind: Graph
defaults:
  unsupported: value
"#,
        GraphFormat::Yaml,
    )
    .expect_err("unknown defaults fields should be rejected");
    assert!(err.to_string().contains("unsupported"));
}

#[test]
fn design_section_83_example_parses_without_validation() {
    let input = r#"
apiVersion: dg/v1
kind: Graph
defaults: { device: { kind: OpenVino, id: 0 }, precision: fp16 }
vars: { model_dir: /opt/models }
nodes:
  - name: cam0
    type: rtsp_source
    params: { url: "rtsp://..." }
  - name: decode
    type: decode
    threads: 4
    params: { hw: auto, zero_copy: true }
  - name: yolo
    type: yolov8
    backend: openvino
    precision: int8
    params: { model: "${model_dir}/yolov8.xml" }
  - name: track
    type: bytetrack
  - name: osd
    type: osd
  - name: enc
    type: encode
    sink: true
edges:
  - cam0.out    -> decode.in
  - decode.image-> yolo.image
  - yolo.dets   -> track.dets
  - track.out   -> osd.in
  - osd.out     -> enc.in
"#;
    let spec = GraphSpec::from_str_with_format(input, GraphFormat::Yaml)
        .expect("the canonical design example should parse");
    assert_eq!(spec.nodes[1].threads, Some(4));
    assert!(spec.nodes[5].sink);
    assert_eq!(spec.connections.len(), 5);
}

#[test]
fn standard_aliases_match_legacy_field_names() {
    let legacy = r#"
apiVersion: dg/v1
kind: Graph
variables: { value: 1 }
nodes:
  - name: source
    kind: source
    params: { count: 1, shape: [1, 4] }
connections: ["source.out -> sink.in"]
"#;
    let standard = r#"
apiVersion: dg/v1
kind: Graph
vars: { value: 1 }
nodes:
  - name: source
    type: source
    params: { count: 1, shape: [1, 4] }
edges: ["source.out -> sink.in"]
"#;
    let legacy = GraphSpec::from_str_with_format(legacy, GraphFormat::Yaml).unwrap();
    let standard = GraphSpec::from_str_with_format(standard, GraphFormat::Yaml).unwrap();
    assert_eq!(legacy, standard);
}

#[test]
fn node_fields_override_defaults_but_not_explicit_params() {
    let spec = GraphSpec {
        defaults: DefaultsSpec {
            backend: Some("mock".to_string()),
            device: Some(DeviceDefault::Named("cpu".to_string())),
            precision: Some("f32".to_string()),
        },
        nodes: vec![
            NodeSpec {
                name: "source".to_string(),
                kind: "source".to_string(),
                params: json!({"count": 1, "shape": [1, 4]}),
                ..NodeSpec::default()
            },
            NodeSpec {
                name: "infer".to_string(),
                kind: "inference".to_string(),
                backend: Some("mock".to_string()),
                device: Some("cpu".to_string()),
                precision: Some("f32".to_string()),
                params: json!({"precision": "f16"}),
                ..NodeSpec::default()
            },
        ],
        connections: vec!["source.out -> infer.in".to_string()],
        ..GraphSpec::default()
    }
    .normalize_with_base_dir(None)
    .expect("node-level fields should normalize");
    let params = &spec.nodes[1].params;
    assert_eq!(params["backend"], "mock");
    assert_eq!(params["device"], "cpu");
    assert_eq!(params["precision"], "f16");
}

#[test]
fn structured_device_default_is_accepted_but_not_injected() {
    let parsed = GraphSpec::from_str_with_format(
        r#"
apiVersion: dg/v1
kind: Graph
defaults:
  device:
    kind: OpenVino
    id: 0
  backend: mock
nodes:
  - name: source
    kind: source
    params: { count: 1, shape: [1, 4] }
  - name: infer
    kind: inference
    params: {}
connections: ["source.out -> infer.in"]
"#,
        GraphFormat::Yaml,
    )
    .expect("structured device default should parse");
    assert!(matches!(
        parsed.defaults.device,
        Some(DeviceDefault::Detailed(_))
    ));
    let normalized = parsed
        .normalize_with_base_dir(None)
        .expect("structured device default should not break normalization");
    assert_eq!(normalized.nodes[1].params["backend"], "mock");
    assert!(normalized.nodes[1].params.get("device").is_none());
}

#[test]
fn standard_fields_still_reject_unknown_node_and_top_level_fields() {
    let node_error = GraphSpec::from_str_with_format(
        r#"
apiVersion: dg/v1
kind: Graph
nodes:
  - name: source
    kind: source
    unknown: true
"#,
        GraphFormat::Yaml,
    )
    .expect_err("unknown node field should be rejected");
    assert!(node_error.to_string().contains("unknown"));

    let top_level_error = GraphSpec::from_str_with_format(
        r#"
apiVersion: dg/v1
kind: Graph
unknown: true
"#,
        GraphFormat::Yaml,
    )
    .expect_err("unknown top-level field should be rejected");
    assert!(top_level_error.to_string().contains("unknown"));
}

#[test]
fn graph_spec_rejects_hanging_references() {
    let spec = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            template: None,
            params: json!({"count": 1, "shape": [1, 4]}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
            ..NodeSpec::default()
        })
        .connect("source.out -> missing.in")
        .build();
    assert!(spec.is_err());
}

#[test]
fn graph_spec_rejects_missing_required_input() {
    let err = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
            ..NodeSpec::default()
        })
        .build()
        .expect_err("unconnected required input should fail");
    let message = err.to_string();
    assert!(message.contains("nodes[sink].ports[in]"));
    assert!(message.contains("required input port sink.in has no incoming connection"));
}

#[test]
fn graph_spec_rejects_multiple_incoming_edges() {
    let err = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "source_a".to_string(),
            kind: "source".to_string(),
            template: None,
            params: json!({"count": 1, "shape": [1, 4]}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "source_b".to_string(),
            kind: "source".to_string(),
            template: None,
            params: json!({"count": 1, "shape": [1, 4]}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
            ..NodeSpec::default()
        })
        .connect("source_a.out -> sink.in")
        .connect("source_b.out -> sink.in")
        .build()
        .expect_err("multiple incoming edges should fail");
    let message = err.to_string();
    assert!(message.contains("connections[1]"));
    assert!(message.contains("input port sink.in already has an incoming connection"));
}

#[test]
fn graph_spec_rejects_duplicate_edges() {
    let err = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            template: None,
            params: json!({"count": 1, "shape": [1, 4]}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
            ..NodeSpec::default()
        })
        .connect("source.out -> sink.in")
        .connect("source.out -> sink.in")
        .build()
        .expect_err("duplicate edges should fail");
    let message = err.to_string();
    assert!(message.contains("connections[1]"));
    assert!(message.contains("duplicate connection"));
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
