use std::collections::HashMap;

use dg_core::{CpuDevice, DataFormat, DataType, DeviceKind, Shape, Tensor, TensorDesc};
use dg_graph::{Graph, GraphSpecBuilder, NodeSpec};
use serde_json::json;

use dg_elements as _;

#[test]
fn yolo_pipeline_emits_nms_filtered_detections() {
    let spec = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "input".to_string(),
            kind: "input".to_string(),
            template: None,
            params: json!({}),
        })
        .add_node(NodeSpec {
            name: "pre".to_string(),
            kind: "yolo_preprocess".to_string(),
            template: None,
            params: json!({
                "input_width": 4,
                "input_height": 4
            }),
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
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
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
        })
        .add_node(NodeSpec {
            name: "distributor".to_string(),
            kind: "distributor".to_string(),
            template: None,
            params: json!({"strategy": "round_robin"}),
        })
        .add_node(NodeSpec {
            name: "infer0".to_string(),
            kind: "mock_inference".to_string(),
            template: None,
            params: json!({"shape": [1, 4], "echo_inputs": true}),
        })
        .add_node(NodeSpec {
            name: "infer1".to_string(),
            kind: "mock_inference".to_string(),
            template: None,
            params: json!({"shape": [1, 4], "echo_inputs": true}),
        })
        .add_node(NodeSpec {
            name: "converger".to_string(),
            kind: "converger".to_string(),
            template: None,
            params: json!({}),
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
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
