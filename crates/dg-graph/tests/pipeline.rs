use std::collections::HashMap;

use dg_core::{CpuDevice, DataFormat, DataType, DeviceKind, Shape, Tensor, TensorDesc};
use dg_graph::{ExecutionSpec, Graph, GraphDiff, GraphSpecBuilder, NodeSpec, ParallelType};
use serde_json::json;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

#[test]
fn source_mock_sink_pipeline_runs_end_to_end() {
    let spec = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            template: None,
            params: json!({
                "count": 2,
                "shape": [1, 4],
                "start": 3.0
            }),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "infer".to_string(),
            kind: "mock_inference".to_string(),
            template: None,
            params: json!({
                "shape": [1, 4],
                "echo_inputs": true
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
        .expect("build pipeline spec");

    let report = Graph::new(spec)
        .expect("build graph")
        .run()
        .expect("run graph");
    let tensors = report.sinks.get("sink").expect("sink outputs");
    assert_eq!(tensors.len(), 2);
    let first_bytes = tensors[0].buffer().read_bytes();
    let second_bytes = tensors[1].buffer().read_bytes();
    assert_eq!(first_bytes.len(), 16);
    assert_eq!(first_bytes, f32_bytes(&[3.0, 3.0, 3.0, 3.0]));
    assert_eq!(second_bytes, f32_bytes(&[4.0, 4.0, 4.0, 4.0]));
}

#[test]
fn injected_input_mock_sink_pipeline_runs_end_to_end() {
    let spec = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "input".to_string(),
            kind: "input".to_string(),
            template: None,
            params: json!({}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "infer".to_string(),
            kind: "mock_inference".to_string(),
            template: None,
            params: json!({
                "shape": [1, 4],
                "echo_inputs": true
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
        .connect("input.out -> infer.in")
        .connect("infer.out -> sink.in")
        .build()
        .expect("build injected input spec");

    let device = CpuDevice::new();
    let desc = TensorDesc::new(
        Shape::new([1, 4]),
        DataType::F32,
        DataFormat::NC,
        DeviceKind::Cpu,
    );
    let tensor = Tensor::allocate(&device, desc).expect("allocate injected tensor");
    tensor
        .buffer()
        .write_from_slice(&f32_bytes(&[1.0, 2.0, 3.0, 4.0]))
        .expect("write injected tensor");

    let report = Graph::new(spec)
        .expect("build graph")
        .run_with_inputs(HashMap::from([("input".to_string(), vec![tensor])]))
        .expect("run graph with input");
    let tensors = report.sinks.get("sink").expect("sink outputs");
    assert_eq!(tensors.len(), 1);
    assert_eq!(
        tensors[0].buffer().read_bytes(),
        f32_bytes(&[1.0, 2.0, 3.0, 4.0])
    );
}

#[test]
fn pipeline_load_balances_packets_across_threaded_element_instances() {
    let count = 32;
    let spec = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            template: None,
            params: json!({
                "count": count,
                "shape": [1, 4],
                "start": 10.0
            }),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "infer".to_string(),
            kind: "mock_inference".to_string(),
            threads: Some(2),
            template: None,
            params: json!({
                "shape": [1, 4],
                "echo_inputs": true
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
        .expect("build threaded pipeline spec");

    let report = Graph::new(spec)
        .expect("build graph")
        .run()
        .expect("run threaded pipeline");
    let tensors = report.sinks.get("sink").expect("sink outputs");
    assert_eq!(tensors.len(), count);
    let mut observed = tensors
        .iter()
        .map(|tensor| tensor.buffer().read_bytes())
        .collect::<Vec<_>>();
    observed.sort();
    let mut expected = (0..count)
        .map(|index| f32_bytes(&[(10 + index) as f32; 4]))
        .collect::<Vec<_>>();
    expected.sort();
    assert_eq!(observed, expected);
}

#[test]
fn pipeline_rejects_multi_instanced_special_handles_at_build_time() {
    let source_spec = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            threads: Some(2),
            params: json!({"count": 1, "shape": [1, 4]}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            params: json!({}),
            ..NodeSpec::default()
        })
        .connect("source.out -> sink.in")
        .build()
        .expect("source graph should validate before runtime build");
    let source_error = Graph::new(source_spec)
        .expect("source graph should construct")
        .run()
        .expect_err("source elements cannot be multi-instanced");
    assert!(source_error.to_string().contains("source elements"));

    let input_spec = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "input".to_string(),
            kind: "input".to_string(),
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
        .connect("input.out -> sink.in")
        .build()
        .expect("input graph should validate before runtime build");
    let input_error = Graph::new(input_spec)
        .expect("input graph should construct")
        .run()
        .expect_err("input handles cannot be multi-instanced");
    assert!(input_error
        .to_string()
        .contains("cannot be multi-instanced"));

    let sink_spec = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            params: json!({"count": 1, "shape": [1, 4]}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            threads: Some(2),
            params: json!({}),
            ..NodeSpec::default()
        })
        .connect("source.out -> sink.in")
        .build()
        .expect("sink graph should validate before runtime build");
    let sink_error = Graph::new(sink_spec)
        .expect("sink graph should construct")
        .run()
        .expect_err("sink handles cannot be multi-instanced");
    assert!(sink_error.to_string().contains("cannot be multi-instanced"));
}

#[test]
fn running_graph_replaces_only_affected_worker_and_rejects_invalid_diff_atomically() {
    let spec = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            params: json!({"count": 128, "shape": [1, 4], "start": 1.0}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "infer".to_string(),
            kind: "mock_inference".to_string(),
            params: json!({"shape": [1, 4], "echo_inputs": true}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            params: json!({}),
            ..NodeSpec::default()
        })
        .connect("source.out -> infer.in")
        .connect("infer.out -> sink.in")
        .build()
        .expect("build running graph spec");
    let graph = Graph::new(spec).expect("construct running graph");
    let mut running = graph.start(HashMap::new()).expect("start running graph");

    let replacement = NodeSpec {
        name: "infer".to_string(),
        kind: "mock_inference".to_string(),
        params: json!({"shape": [1, 4], "echo_inputs": false, "fill_value": 7}),
        ..NodeSpec::default()
    };
    running
        .apply_hot_update(GraphDiff {
            updated_nodes: vec![replacement],
            ..GraphDiff::default()
        })
        .expect("replace running worker");

    let invalid = GraphDiff {
        added_nodes: vec![NodeSpec {
            name: "broken".to_string(),
            kind: "does_not_exist".to_string(),
            params: json!({}),
            ..NodeSpec::default()
        }],
        ..GraphDiff::default()
    };
    assert!(running.apply_hot_update(invalid).is_err());

    let report = running.finish().expect("finish updated graph");
    let outputs = report.sinks.get("sink").expect("sink outputs");
    assert!(!outputs.is_empty());
    assert!(outputs
        .iter()
        .any(|tensor| { tensor.buffer().read_bytes().iter().all(|byte| *byte == 7) }));
}

#[test]
fn hot_update_keeps_unaffected_branch_lossless_under_backpressure() {
    let count = 2_048;
    let execution = ExecutionSpec {
        parallel: ParallelType::Pipeline,
        queue_capacity: 1,
        workers: None,
    };
    for iteration in 0..16 {
        let spec = GraphSpecBuilder::new()
            .execution(execution.clone())
            .add_node(NodeSpec {
                name: "source".to_string(),
                kind: "source".to_string(),
                params: json!({"count": count, "shape": [1, 4], "start": 1.0}),
                ..NodeSpec::default()
            })
            .add_node(NodeSpec {
                name: "affected".to_string(),
                kind: "mock_inference".to_string(),
                params: json!({"shape": [1, 4], "echo_inputs": true}),
                ..NodeSpec::default()
            })
            .add_node(NodeSpec {
                name: "unaffected".to_string(),
                kind: "sink".to_string(),
                params: json!({}),
                ..NodeSpec::default()
            })
            .add_node(NodeSpec {
                name: "affected_sink".to_string(),
                kind: "sink".to_string(),
                params: json!({}),
                ..NodeSpec::default()
            })
            .connect("source.out -> affected.in")
            .connect("source.out -> unaffected.in")
            .connect("affected.out -> affected_sink.in")
            .build()
            .expect("build hot-update stress graph");
        let graph = Graph::new(spec).expect("construct hot-update stress graph");
        let mut running = graph.start(HashMap::new()).expect("start stress graph");
        std::thread::sleep(std::time::Duration::from_millis(1));
        running
            .apply_hot_update(GraphDiff {
                updated_nodes: vec![NodeSpec {
                    name: "affected".to_string(),
                    kind: "mock_inference".to_string(),
                    params: json!({
                        "shape": [1, 4],
                        "echo_inputs": false,
                        "fill_value": 7
                    }),
                    ..NodeSpec::default()
                }],
                ..GraphDiff::default()
            })
            .unwrap_or_else(|error| panic!("iteration {iteration} hot update failed: {error}"));
        let report = running
            .finish()
            .unwrap_or_else(|error| panic!("iteration {iteration} did not finish: {error}"));
        let outputs = report
            .sinks
            .get("unaffected")
            .expect("unaffected sink outputs");
        assert_eq!(
            outputs.len(),
            count,
            "iteration {iteration} lost packets on unaffected branch"
        );
    }
}
