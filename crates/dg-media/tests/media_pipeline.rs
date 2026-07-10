//! End-to-end pipeline tests for the media elements using mock/recorded
//! in-memory frames: input -> decode -> resize -> osd -> encode -> sink.

use std::collections::HashMap;

use dg_core::{CpuDevice, DataFormat, DataType, DeviceKind, Shape, Tensor, TensorDesc};
use dg_graph::{Graph, GraphSpecBuilder, NodeSpec};
use serde_json::json;

use dg_media as _;

fn recorded_frame_tensor(bytes: Vec<u8>) -> Tensor {
    let device = CpuDevice::new();
    let desc = TensorDesc::new(
        Shape::new([bytes.len()]),
        DataType::U8,
        DataFormat::N,
        DeviceKind::Cpu,
    );
    let tensor = Tensor::allocate(&device, desc).expect("allocate recorded frame");
    tensor
        .buffer()
        .write_from_slice(&bytes)
        .expect("write recorded frame");
    tensor
}

fn node(name: &str, kind: &str, params: serde_json::Value) -> NodeSpec {
    NodeSpec {
        name: name.to_string(),
        kind: kind.to_string(),
        template: None,
        params,
    }
}

#[test]
fn decode_resize_osd_encode_pipeline_runs_end_to_end() {
    let spec = GraphSpecBuilder::new()
        .add_node(node("input", "input", json!({})))
        .add_node(node(
            "decode",
            "media_decode",
            json!({ "width": 2, "height": 2, "channels": 1 }),
        ))
        .add_node(node(
            "resize",
            "media_resize",
            json!({ "width": 4, "height": 4 }),
        ))
        .add_node(node(
            "osd",
            "media_osd",
            json!({
                "boxes": [{ "x": 0, "y": 0, "width": 4, "height": 4 }],
                "color": [255],
                "thickness": 1
            }),
        ))
        .add_node(node("encode", "media_encode", json!({})))
        .add_node(node("sink", "sink", json!({})))
        .connect("input.out -> decode.in")
        .connect("decode.out -> resize.in")
        .connect("resize.out -> osd.in")
        .connect("osd.out -> encode.in")
        .connect("encode.out -> sink.in")
        .build()
        .expect("build media pipeline spec");

    let recorded = vec![
        recorded_frame_tensor(vec![10, 20, 30, 40]),
        recorded_frame_tensor(vec![1, 2, 3, 4]),
    ];

    let report = Graph::new(spec)
        .expect("build graph")
        .run_with_inputs(HashMap::from([("input".to_string(), recorded)]))
        .expect("run media pipeline");
    let outputs = report.sinks.get("sink").expect("sink outputs");
    assert_eq!(outputs.len(), 2);

    for (index, interior) in [(0_usize, 10_u8), (1_usize, 1_u8)] {
        let bytes = outputs[index].buffer().read_bytes();
        assert_eq!(bytes.len(), 16);
        // Border pixels overwritten by the OSD box.
        assert_eq!(bytes[0], 255);
        assert_eq!(bytes[3], 255);
        assert_eq!(bytes[12], 255);
        assert_eq!(bytes[15], 255);
        // Interior pixel keeps the nearest-neighbour resized value.
        assert_eq!(bytes[5], interior);
    }
}

#[test]
fn decode_pipeline_rejects_wrong_payload_size() {
    let spec = GraphSpecBuilder::new()
        .add_node(node("input", "input", json!({})))
        .add_node(node(
            "decode",
            "media_decode",
            json!({ "width": 4, "height": 4, "channels": 3 }),
        ))
        .add_node(node("sink", "sink", json!({})))
        .connect("input.out -> decode.in")
        .connect("decode.out -> sink.in")
        .build()
        .expect("build decode spec");

    let result = Graph::new(spec)
        .expect("build graph")
        .run_with_inputs(HashMap::from([(
            "input".to_string(),
            vec![recorded_frame_tensor(vec![0; 7])],
        )]));
    let err = result.expect_err("expected decode failure");
    assert!(err.to_string().contains("media_decode"));
}

#[test]
fn media_elements_are_registered() {
    for kind in ["media_decode", "media_encode", "media_resize", "media_osd"] {
        assert!(
            dg_graph::find_element(kind).is_some(),
            "element {kind} must be registered"
        );
    }
}
