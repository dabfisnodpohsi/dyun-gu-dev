use std::collections::HashMap;

use dg_core::{CpuDevice, DataFormat, DataType, DeviceKind, Shape, Tensor, TensorDesc};
use dg_graph::{Graph, GraphSpecBuilder, NodeSpec};
use serde_json::json;

use dg_elements as _;

fn node(kind: &str, params: serde_json::Value) -> NodeSpec {
    NodeSpec {
        name: "algorithm".to_string(),
        kind: kind.to_string(),
        template: None,
        params,
        ..NodeSpec::default()
    }
}

#[test]
fn algorithm_element_parameters_are_validated_at_load_time() {
    let invalid = [
        (
            "yolo_preprocess",
            json!({"input_width": 640, "unknown": true}),
            "unknown field `unknown`",
        ),
        (
            "yolo_postprocess",
            json!({"confidence_threshold": 1.1}),
            "field confidence_threshold must be between 0 and 1",
        ),
        (
            "resnet_preprocess",
            json!({"std": [0.229, 0.0, 0.225]}),
            "field std values must be greater than zero",
        ),
        (
            "resnet_postprocess",
            json!({"labels": ["valid", 2]}),
            "field labels must contain strings",
        ),
        (
            "retinaface",
            json!({"stride": 0}),
            "field stride must be non-zero",
        ),
        (
            "bytetrack",
            json!({"match_iou": 1.1}),
            "field match_iou must be between 0 and 1",
        ),
        (
            "ppocr_det",
            json!({"threshold": -0.1}),
            "field threshold must be between 0 and 1",
        ),
        (
            "ppocr_rec",
            json!({"alphabet": ""}),
            "field alphabet must not be empty",
        ),
        (
            "ppocr_rec",
            json!({"alphabet": "ab", "blank_index": 3}),
            "field blank_index must not exceed the alphabet length (2)",
        ),
        (
            "distributor",
            json!({"strategy": 1}),
            "field strategy must be a string",
        ),
        (
            "converger",
            json!({"strategy": "round_robin"}),
            "unknown field `strategy`",
        ),
        (
            "filter",
            json!({"unknown": true}),
            "unknown field `unknown`",
        ),
        (
            "filter",
            json!({"mode": "invalid"}),
            "field mode must be one of pass, drop",
        ),
        ("filter", json!({"mode": 1}), "field mode must be a string"),
    ];

    for (kind, params, expected) in invalid {
        let err = GraphSpecBuilder::new()
            .add_node(node(kind, params))
            .build()
            .expect_err("invalid algorithm params must fail during graph loading");
        let message = err.to_string();
        assert!(message.contains("nodes[algorithm].params"), "{message}");
        assert!(message.contains(expected), "{message}");
    }
}

#[test]
fn yolo_pipeline_emits_nms_filtered_detections() {
    let spec = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "input".to_string(),
            kind: "input".to_string(),
            template: None,
            params: json!({}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "pre".to_string(),
            kind: "yolo_preprocess".to_string(),
            template: None,
            params: json!({
                "input_width": 4,
                "input_height": 4
            }),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "infer".to_string(),
            kind: "mock_inference".to_string(),
            template: None,
            params: json!({
                "shape": [1, 3, 4, 4],
                "output_shape": [1, 2, 6],
                "echo_inputs": false,
                "fill_value": 0
            }),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "post".to_string(),
            kind: "yolo_postprocess".to_string(),
            template: None,
            params: json!({
                "input_width": 4,
                "input_height": 4,
                "class_count": 1,
                "confidence_threshold": 0.2,
                "nms_threshold": 0.4
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
        .connect("input.out -> pre.in")
        .connect("pre.out -> infer.in")
        .connect("infer.out -> post.in")
        .connect("post.out -> sink.in")
        .build()
        .expect("build yolo graph");

    let device = CpuDevice::new();
    let input_desc = TensorDesc::new(
        Shape::new([1, 3, 2, 4]),
        DataType::U8,
        DataFormat::NCHW,
        DeviceKind::Cpu,
    );
    let input = Tensor::allocate(&device, input_desc).expect("allocate image");
    input
        .buffer()
        .write_from_slice(&[255; 24])
        .expect("write image");

    let report = Graph::new(spec)
        .expect("build graph")
        .run_with_inputs(HashMap::from([("input".to_string(), vec![input])]))
        .expect("run yolo graph");
    let detections = report.detections.get("sink").expect("detection output");
    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0].class_id, 0);
    assert!((0.0..=4.0).contains(&detections[0].bbox.x));
    assert!((0.0..=2.0).contains(&detections[0].bbox.y));
    assert!(detections[0].bbox.w <= 4.0);
    assert!(detections[0].bbox.h <= 2.0);
}

#[test]
fn distributor_and_converger_preserve_all_packets() {
    let spec = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "source".to_string(),
            kind: "source".to_string(),
            template: None,
            params: json!({"count": 4, "shape": [1, 4]}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "distributor".to_string(),
            kind: "distributor".to_string(),
            template: None,
            params: json!({"strategy": "round_robin"}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "infer0".to_string(),
            kind: "mock_inference".to_string(),
            template: None,
            params: json!({"shape": [1, 4], "echo_inputs": true}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "infer1".to_string(),
            kind: "mock_inference".to_string(),
            template: None,
            params: json!({"shape": [1, 4], "echo_inputs": true}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "converger".to_string(),
            kind: "converger".to_string(),
            template: None,
            params: json!({}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
            ..NodeSpec::default()
        })
        .connect("source.out -> distributor.in")
        .connect("distributor.out0 -> infer0.in")
        .connect("distributor.out1 -> infer1.in")
        .connect("infer0.out -> converger.in0")
        .connect("infer1.out -> converger.in1")
        .connect("converger.out -> sink.in")
        .build()
        .expect("build parallel graph");

    let report = Graph::new(spec)
        .expect("build graph")
        .run()
        .expect("run graph");
    assert_eq!(report.sinks.get("sink").expect("sink output").len(), 4);
}

#[test]
fn filter_pass_and_drop_modes_control_packet_forwarding() {
    for (mode, expected_packets) in [("pass", 4), ("drop", 0)] {
        let spec = GraphSpecBuilder::new()
            .add_node(NodeSpec {
                name: "source".to_string(),
                kind: "source".to_string(),
                template: None,
                params: json!({"count": 4, "shape": [1, 4]}),
                ..NodeSpec::default()
            })
            .add_node(NodeSpec {
                name: "filter".to_string(),
                kind: "filter".to_string(),
                template: None,
                params: json!({"mode": mode}),
                ..NodeSpec::default()
            })
            .add_node(NodeSpec {
                name: "sink".to_string(),
                kind: "sink".to_string(),
                template: None,
                params: json!({}),
                ..NodeSpec::default()
            })
            .connect("source.out -> filter.in")
            .connect("filter.out -> sink.in")
            .build()
            .expect("build filter graph");

        let report = Graph::new(spec)
            .expect("build graph")
            .run()
            .expect("run filter graph");
        assert_eq!(
            report.sinks.get("sink").expect("sink output").len(),
            expected_packets
        );
    }
}

#[test]
fn converger_allows_subset_of_input_ports() {
    GraphSpecBuilder::new()
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
            name: "converger".to_string(),
            kind: "converger".to_string(),
            template: None,
            params: json!({}),
            ..NodeSpec::default()
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
            ..NodeSpec::default()
        })
        .connect("source_a.out -> converger.in0")
        .connect("source_b.out -> converger.in2")
        .connect("converger.out -> sink.in")
        .build()
        .expect("a subset of converger inputs should be valid");
}
