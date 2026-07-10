use dg_graph::{
    ExecutionSpec, Graph, GraphFormat, GraphSpec, GraphSpecBuilder, NodeSpec, ParallelType,
};
use proptest::prelude::*;
use serde_json::json;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn linear_spec(execution: ExecutionSpec, count: usize) -> GraphSpec {
    GraphSpecBuilder::new()
        .execution(execution)
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            template: None,
            params: json!({"count": count, "shape": [1, 4], "start": 1.0}),
        })
        .add_node(NodeSpec {
            name: "infer".to_string(),
            kind: "mock_inference".to_string(),
            template: None,
            params: json!({"shape": [1, 4], "echo_inputs": true}),
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
        .expect("build linear spec")
}

fn branched_spec(execution: ExecutionSpec, count: usize) -> GraphSpec {
    GraphSpecBuilder::new()
        .execution(execution)
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            template: None,
            params: json!({"count": count, "shape": [1, 4], "start": 1.0}),
        })
        .add_node(NodeSpec {
            name: "infer_a".to_string(),
            kind: "mock_inference".to_string(),
            template: None,
            params: json!({"shape": [1, 4], "echo_inputs": true}),
        })
        .add_node(NodeSpec {
            name: "infer_b".to_string(),
            kind: "mock_inference".to_string(),
            template: None,
            params: json!({"shape": [1, 4], "echo_inputs": true}),
        })
        .add_node(NodeSpec {
            name: "sink_a".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
        })
        .add_node(NodeSpec {
            name: "sink_b".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
        })
        .connect("source.out -> infer_a.in")
        .connect("source.out -> infer_b.in")
        .connect("infer_a.out -> sink_a.in")
        .connect("infer_b.out -> sink_b.in")
        .build()
        .expect("build branched spec")
}

fn assert_in_order(report: &dg_graph::GraphReport, sink: &str, count: usize) {
    let tensors = report.sinks.get(sink).expect("sink outputs");
    assert_eq!(tensors.len(), count);
    for (index, tensor) in tensors.iter().enumerate() {
        let expected = 1.0 + index as f32;
        assert_eq!(
            tensor.buffer().read_bytes(),
            f32_bytes(&[expected; 4]),
            "packet {index} out of order in {sink}"
        );
    }
}

#[test]
fn pipeline_backpressure_preserves_order_with_tiny_queues() {
    let execution = ExecutionSpec {
        parallel: ParallelType::Pipeline,
        queue_capacity: 1,
        workers: None,
    };
    let report = Graph::new(linear_spec(execution, 64))
        .expect("build graph")
        .run()
        .expect("run graph");
    assert_in_order(&report, "sink", 64);
}

#[test]
fn pipeline_branches_deliver_all_packets_under_backpressure() {
    let execution = ExecutionSpec {
        parallel: ParallelType::Pipeline,
        queue_capacity: 2,
        workers: None,
    };
    let report = Graph::new(branched_spec(execution, 32))
        .expect("build graph")
        .run()
        .expect("run graph");
    assert_in_order(&report, "sink_a", 32);
    assert_in_order(&report, "sink_b", 32);
}

#[test]
fn sequential_execution_runs_end_to_end() {
    let execution = ExecutionSpec {
        parallel: ParallelType::Sequential,
        ..ExecutionSpec::default()
    };
    let report = Graph::new(branched_spec(execution, 40))
        .expect("build graph")
        .run()
        .expect("run graph");
    assert_in_order(&report, "sink_a", 40);
    assert_in_order(&report, "sink_b", 40);
}

#[test]
fn task_execution_runs_end_to_end() {
    let execution = ExecutionSpec {
        parallel: ParallelType::Task,
        workers: Some(2),
        ..ExecutionSpec::default()
    };
    let report = Graph::new(branched_spec(execution, 40))
        .expect("build graph")
        .run()
        .expect("run graph");
    assert_in_order(&report, "sink_a", 40);
    assert_in_order(&report, "sink_b", 40);
}

fn failing_spec(execution: ExecutionSpec) -> GraphSpec {
    GraphSpecBuilder::new()
        .execution(execution)
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            template: None,
            // f16 sources are rejected at run time, exercising error paths.
            params: json!({"count": 4, "shape": [1, 4], "dtype": "f16"}),
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
        })
        .connect("source.out -> sink.in")
        .build()
        .expect("build failing spec")
}

#[test]
fn element_errors_propagate_in_every_parallel_mode() {
    for parallel in [
        ParallelType::Sequential,
        ParallelType::Task,
        ParallelType::Pipeline,
    ] {
        let execution = ExecutionSpec {
            parallel,
            ..ExecutionSpec::default()
        };
        let result = Graph::new(failing_spec(execution))
            .expect("build graph")
            .run();
        assert!(result.is_err(), "expected error in {parallel:?} mode");
    }
}

#[test]
fn validation_rejects_invalid_execution_settings() {
    let zero_capacity = GraphSpecBuilder::new()
        .execution(ExecutionSpec {
            queue_capacity: 0,
            ..ExecutionSpec::default()
        })
        .build();
    assert!(zero_capacity.is_err());

    let zero_workers = GraphSpecBuilder::new()
        .execution(ExecutionSpec {
            parallel: ParallelType::Task,
            workers: Some(0),
            ..ExecutionSpec::default()
        })
        .build();
    assert!(zero_workers.is_err());

    let pipeline_workers = GraphSpecBuilder::new()
        .execution(ExecutionSpec {
            parallel: ParallelType::Pipeline,
            workers: Some(4),
            ..ExecutionSpec::default()
        })
        .build();
    assert!(pipeline_workers.is_err());
}

#[test]
fn execution_spec_defaults_apply_when_omitted() {
    let spec =
        GraphSpec::from_str_with_format("apiVersion: dg/v1\nkind: Graph\n", GraphFormat::Yaml)
            .expect("parse minimal spec");
    assert_eq!(spec.execution, ExecutionSpec::default());
    assert_eq!(spec.execution.parallel, ParallelType::Pipeline);
    assert_eq!(
        spec.execution.queue_capacity,
        dg_graph::DEFAULT_QUEUE_CAPACITY
    );
}

#[test]
fn json_schema_export_describes_execution_model() {
    let schema = GraphSpec::json_schema().expect("export schema");
    let value: serde_json::Value = serde_json::from_str(&schema).expect("schema is valid JSON");
    let properties = value
        .get("properties")
        .and_then(|properties| properties.as_object())
        .expect("schema has properties");
    for field in ["apiVersion", "execution", "nodes", "connections"] {
        assert!(properties.contains_key(field), "schema missing {field}");
    }
    assert!(schema.contains("queue_capacity"));
    assert!(schema.contains("pipeline"));
}

proptest! {
    #[test]
    fn execution_spec_round_trips_across_formats(
        parallel_index in 0_usize..3,
        queue_capacity in 1_usize..128,
        workers in proptest::option::of(1_usize..16),
    ) {
        let parallel = [
            ParallelType::Sequential,
            ParallelType::Task,
            ParallelType::Pipeline,
        ][parallel_index];
        let workers = if parallel == ParallelType::Task { workers } else { None };
        let execution = ExecutionSpec { parallel, queue_capacity, workers };
        let spec = linear_spec(execution, 1);
        for format in [GraphFormat::Yaml, GraphFormat::Json, GraphFormat::Toml] {
            let encoded = spec.to_string_with_format(format).expect("serialize spec");
            let decoded = GraphSpec::from_str_with_format(&encoded, format).expect("parse spec");
            prop_assert_eq!(&decoded, &spec);
        }
    }
}
