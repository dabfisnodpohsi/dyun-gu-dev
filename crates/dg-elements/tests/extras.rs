use std::collections::HashMap;

use dg_core::{CpuDevice, DataFormat, DataType, DeviceKind, Shape, Tensor, TensorDesc};
use dg_graph::{Graph, GraphSpecBuilder, NodeSpec};
use serde_json::json;

use dg_elements as _;

fn tensor(shape: impl Into<Vec<usize>>, values: &[f32]) -> Tensor {
    let device = CpuDevice::new();
    let tensor = Tensor::allocate(
        &device,
        TensorDesc::new(
            Shape::new(shape),
            DataType::F32,
            DataFormat::NC,
            DeviceKind::Cpu,
        ),
    )
    .expect("allocate test tensor");
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    tensor
        .buffer()
        .write_from_slice(&bytes)
        .expect("write tensor");
    tensor
}

fn graph_with_input(kind: &str, params: serde_json::Value) -> dg_graph::GraphSpec {
    GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "input".to_string(),
            kind: "input".to_string(),
            template: None,
            params: json!({}),
        })
        .add_node(NodeSpec {
            name: "algorithm".to_string(),
            kind: kind.to_string(),
            template: None,
            params,
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
        })
        .connect("input.out -> algorithm.in")
        .connect("algorithm.out -> sink.in")
        .build()
        .expect("build algorithm graph")
}

#[test]
fn resnet_postprocess_orders_top_k_results() {
    let graph = Graph::new(graph_with_input(
        "resnet_postprocess",
        json!({"top_k": 2, "labels": ["zero", "one", "two"]}),
    ))
    .expect("build graph");
    let report = graph
        .run_with_inputs(HashMap::from([(
            "input".to_string(),
            vec![tensor([1, 3], &[1.0, 3.0, 2.0])],
        )]))
        .expect("run graph");
    let results = report
        .classifications
        .get("sink")
        .expect("classification results");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].class_id, 1);
    assert_eq!(results[1].class_id, 2);
    assert!((results[0].score + results[1].score) < 1.0);
}

#[test]
fn retinaface_decodes_face_and_landmarks() {
    let spec = GraphSpecBuilder::new()
        .add_node(NodeSpec {
            name: "input".to_string(),
            kind: "input".to_string(),
            template: None,
            params: json!({}),
        })
        .add_node(NodeSpec {
            name: "infer".to_string(),
            kind: "mock_inference".to_string(),
            template: None,
            params: json!({"shape": [1, 15], "echo_inputs": true}),
        })
        .add_node(NodeSpec {
            name: "face".to_string(),
            kind: "retinaface".to_string(),
            template: None,
            params: json!({
                "image_width": 32,
                "image_height": 32,
                "stride": 16,
                "anchor_sizes": [0.25],
                "confidence_threshold": 0.4
            }),
        })
        .add_node(NodeSpec {
            name: "sink".to_string(),
            kind: "sink".to_string(),
            template: None,
            params: json!({}),
        })
        .connect("input.out -> infer.in")
        .connect("infer.out -> face.in")
        .connect("face.out -> sink.in")
        .build()
        .expect("build retinaface graph");
    let report = Graph::new(spec)
        .expect("build graph")
        .run_with_inputs(HashMap::from([(
            "input".to_string(),
            vec![tensor([1, 15], &[0.0; 15])],
        )]))
        .expect("run graph");
    let faces = report.faces.get("sink").expect("face results");
    assert_eq!(faces.len(), 1);
    assert_eq!(faces[0].landmarks.len(), 5);
    assert!(faces[0]
        .landmarks
        .iter()
        .all(|point| { (0.0..=32.0).contains(&point.x) && (0.0..=32.0).contains(&point.y) }));
}

#[test]
fn ppocr_rec_decodes_ctc_text() {
    let graph = Graph::new(graph_with_input(
        "ppocr_rec",
        json!({"alphabet": "ab", "blank_index": 2}),
    ))
    .expect("build graph");
    let values = [0.9, 0.1, 0.0, 0.8, 0.1, 0.0, 0.1, 0.9, 0.0, 0.1, 0.9, 0.0];
    let report = graph
        .run_with_inputs(HashMap::from([(
            "input".to_string(),
            vec![tensor([4, 3], &values)],
        )]))
        .expect("run graph");
    let results = report.ocr.get("sink").expect("ocr results");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].text, "ab");
}

#[test]
fn ppocr_det_finds_connected_text_region() {
    let graph =
        Graph::new(graph_with_input("ppocr_det", json!({"threshold": 0.5}))).expect("build graph");
    let values = [0.0, 0.8, 0.8, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0];
    let report = graph
        .run_with_inputs(HashMap::from([(
            "input".to_string(),
            vec![tensor([1, 1, 3, 4], &values)],
        )]))
        .expect("run graph");
    let results = report.ocr.get("sink").expect("ocr results");
    assert_eq!(results.len(), 1);
    let bbox = results[0].bbox.expect("detected region");
    assert!(bbox.w > 0.0 && bbox.h > 0.0);
}
