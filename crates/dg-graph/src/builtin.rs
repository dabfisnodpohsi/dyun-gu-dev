use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use dg_core::{DataFormat, DataType, DeviceKind, Shape, Tensor, TensorDesc};
use dg_runtime::{
    BackendKind, BackendOptions, MockOptions, ModelSource, Runtime, RuntimeOption, TensorInfo,
};
use serde_json::{Map, Value};
use tracing::trace;

use crate::element::{
    CreatedElement, Element, ElementHandle, ElementIo, PortSchema, SinkCollector,
};
use crate::error::{Error, Result};
use crate::packet::Packet;
use crate::registry::ElementDescriptor;
use crate::spec::NodeSpec;
use crate::{ParamField, ParamType};

const SOURCE_OUTPUT_PORT: PortSchema = PortSchema {
    name: "out",
    dtype: Some(DataType::F32),
    required: false,
};
const INPUT_OUTPUT_PORT: PortSchema = PortSchema {
    name: "out",
    dtype: Some(DataType::F32),
    required: false,
};
const INFER_INPUT_PORT: PortSchema = PortSchema {
    name: "in",
    dtype: Some(DataType::F32),
    required: true,
};
const INFER_OUTPUT_PORT: PortSchema = PortSchema {
    name: "out",
    dtype: Some(DataType::F32),
    required: false,
};
const SINK_INPUT_PORT: PortSchema = PortSchema {
    name: "in",
    dtype: None,
    required: true,
};
const SOURCE_PARAM_FIELDS: &[&str] = &["count", "shape", "dtype", "format", "start"];
const MOCK_INFERENCE_PARAM_FIELDS: &[&str] = &[
    "shape",
    "output_shape",
    "dtype",
    "output_dtype",
    "echo_inputs",
    "fill_value",
];
const DTYPE_VALUES: &[&str] = &["f32", "f16", "bf16", "u8", "i8", "u16", "i16"];
const FORMAT_VALUES: &[&str] = &[
    "auto", "nchw", "nhwc", "nc", "n", "nc4hw", "nc8hw", "ncdhw", "oihw",
];
const EMPTY_PARAMS: &[ParamField] = &[];
const SOURCE_PARAMS: &[ParamField] = &[
    ParamField {
        name: "count",
        ty: ParamType::Uint,
        required: false,
    },
    ParamField {
        name: "shape",
        ty: ParamType::Array(&ParamType::Uint),
        required: false,
    },
    ParamField {
        name: "dtype",
        ty: ParamType::Enum(DTYPE_VALUES),
        required: false,
    },
    ParamField {
        name: "format",
        ty: ParamType::Enum(FORMAT_VALUES),
        required: false,
    },
    ParamField {
        name: "start",
        ty: ParamType::Float,
        required: false,
    },
];
const MOCK_INFERENCE_PARAMS: &[ParamField] = &[
    ParamField {
        name: "shape",
        ty: ParamType::Array(&ParamType::Uint),
        required: false,
    },
    ParamField {
        name: "output_shape",
        ty: ParamType::Array(&ParamType::Uint),
        required: false,
    },
    ParamField {
        name: "dtype",
        ty: ParamType::Enum(DTYPE_VALUES),
        required: false,
    },
    ParamField {
        name: "output_dtype",
        ty: ParamType::Enum(DTYPE_VALUES),
        required: false,
    },
    ParamField {
        name: "echo_inputs",
        ty: ParamType::Bool,
        required: false,
    },
    ParamField {
        name: "fill_value",
        ty: ParamType::Uint,
        required: false,
    },
];

inventory::submit! {
    ElementDescriptor {
        kind: "input",
        input_ports: &[],
        output_ports: &[INPUT_OUTPUT_PORT],
        params: EMPTY_PARAMS,
        validate: Some(validate_empty_params),
        create: create_input,
    }
}

inventory::submit! {
    ElementDescriptor {
        kind: "source",
        input_ports: &[],
        output_ports: &[SOURCE_OUTPUT_PORT],
        params: SOURCE_PARAMS,
        validate: Some(validate_source),
        create: create_source,
    }
}

inventory::submit! {
    ElementDescriptor {
        kind: "mock_inference",
        input_ports: &[INFER_INPUT_PORT],
        output_ports: &[INFER_OUTPUT_PORT],
        params: MOCK_INFERENCE_PARAMS,
        validate: Some(validate_mock_inference),
        create: create_mock_inference,
    }
}

inventory::submit! {
    ElementDescriptor {
        kind: "sink",
        input_ports: &[SINK_INPUT_PORT],
        output_ports: &[],
        params: EMPTY_PARAMS,
        validate: Some(validate_empty_params),
        create: create_sink,
    }
}

struct SourceElement {
    count: usize,
    shape: Shape,
    dtype: DataType,
    format: DataFormat,
    start: f32,
}

struct InputElement {
    queue: Arc<Mutex<VecDeque<Tensor>>>,
}

impl Element for SourceElement {
    fn run(self: Box<Self>, io: ElementIo) -> Result<()> {
        trace!(node = %io.name, count = self.count, "running source element");
        for index in 0..self.count {
            if io.should_stop() {
                return Err(Error::NotRunning);
            }
            let step = usize_to_exact_f32(index, "source index")?;
            let tensor = filled_tensor(
                self.shape.clone(),
                self.dtype,
                self.format,
                self.start + step,
            )?;
            io.send("out", Packet::tensor(tensor))?;
        }
        io.broadcast_eos()
    }
}

impl Element for InputElement {
    fn run(self: Box<Self>, io: ElementIo) -> Result<()> {
        trace!(node = %io.name, "running input element");
        let mut pending = {
            let mut guard = self
                .queue
                .lock()
                .map_err(|_| Error::Runtime("input queue poisoned".to_string()))?;
            guard.drain(..).collect::<VecDeque<_>>()
        };
        while let Some(tensor) = pending.pop_front() {
            io.send("out", Packet::tensor(tensor))?;
        }
        io.broadcast_eos()
    }
}

struct MockInferenceElement {
    runtime: Runtime,
    echo_inputs: bool,
    fill_value: u8,
}

impl Element for MockInferenceElement {
    fn run(mut self: Box<Self>, io: ElementIo) -> Result<()> {
        trace!(node = %io.name, "running mock inference element");
        loop {
            let packet = match io.recv("in") {
                Ok(Some(packet)) => packet,
                Ok(None) => {
                    if io.should_stop() {
                        return Err(Error::NotRunning);
                    }
                    continue;
                }
                Err(err) => return Err(err),
            };
            if packet.is_eos() {
                io.broadcast_eos()?;
                return Ok(());
            }

            let tensor = packet
                .tensor_ref()
                .ok_or_else(|| Error::Runtime("expected tensor payload".to_string()))?
                .clone();
            let meta = packet.meta.clone();
            let outputs = if self.echo_inputs {
                self.runtime.run(&[tensor])?
            } else {
                let mut outputs = self.runtime.run(&[tensor])?;
                for output in &mut outputs {
                    let bytes = output.buffer().read_bytes();
                    let fill = vec![self.fill_value; bytes.len()];
                    output.buffer().write_from_slice(&fill)?;
                }
                outputs
            };
            for output in outputs {
                io.send("out", Packet::tensor(output).with_meta(meta.clone()))?;
            }
        }
    }
}

struct SinkElement {
    collector: Arc<Mutex<SinkCollector>>,
}

impl Element for SinkElement {
    fn run(self: Box<Self>, io: ElementIo) -> Result<()> {
        trace!(node = %io.name, "running sink element");
        loop {
            let packet = match io.recv("in") {
                Ok(Some(packet)) => packet,
                Ok(None) => {
                    if io.should_stop() {
                        return Err(Error::NotRunning);
                    }
                    continue;
                }
                Err(err) => return Err(err),
            };
            if packet.is_eos() {
                return Ok(());
            }
            let mut guard = self
                .collector
                .lock()
                .map_err(|_| Error::Runtime("sink collector poisoned".to_string()))?;
            if let Some(tensor) = packet.tensor_ref() {
                guard.tensors.push(tensor.clone());
            } else if let Some(detections) = packet.detections_ref() {
                guard.detections.push(detections.to_vec());
            } else if let Some(results) = packet.classifications_ref() {
                guard.classifications.push(results.to_vec());
            } else if let Some(results) = packet.faces_ref() {
                guard.faces.push(results.to_vec());
            } else if let Some(results) = packet.tracks_ref() {
                guard.tracks.push(results.to_vec());
            } else if let Some(results) = packet.ocr_ref() {
                guard.ocr.push(results.to_vec());
            } else {
                return Err(Error::Runtime(
                    "expected tensor or detections payload".to_string(),
                ));
            }
            io.finish_packet()?;
        }
    }
}

fn create_source(node: &NodeSpec) -> Result<CreatedElement> {
    let element = parse_source(node)?;
    Ok(CreatedElement {
        element: Box::new(element),
        handle: ElementHandle::None,
    })
}

fn validate_source(node: &NodeSpec) -> Result<()> {
    parse_source(node).map(|_| ())
}

fn parse_source(node: &NodeSpec) -> Result<SourceElement> {
    let params = params_object(node)?;
    reject_unknown_fields(params, SOURCE_PARAM_FIELDS)?;
    let count = read_usize(params, "count", 1)?;
    let shape = read_shape(params, "shape", &[1, 4])?;
    let dtype = read_dtype(params, "dtype")?.unwrap_or(DataType::F32);
    let format = read_format(params, "format")?.unwrap_or(DataFormat::NC);
    let start = read_f32(params, "start", 0.0)?;
    Ok(SourceElement {
        count,
        shape,
        dtype,
        format,
        start,
    })
}

fn create_input(node: &NodeSpec) -> Result<CreatedElement> {
    validate_empty_params(node)?;
    let queue = Arc::new(Mutex::new(VecDeque::new()));
    Ok(CreatedElement {
        element: Box::new(InputElement {
            queue: queue.clone(),
        }),
        handle: ElementHandle::Input(queue),
    })
}

fn create_mock_inference(node: &NodeSpec) -> Result<CreatedElement> {
    let config = parse_mock_inference(node)?;
    let runtime = Runtime::new(config.option)?;
    Ok(CreatedElement {
        element: Box::new(MockInferenceElement {
            runtime,
            echo_inputs: config.echo_inputs,
            fill_value: config.fill_value,
        }),
        handle: ElementHandle::None,
    })
}

fn validate_mock_inference(node: &NodeSpec) -> Result<()> {
    parse_mock_inference(node).map(|_| ())
}

struct MockInferenceConfig {
    option: RuntimeOption,
    echo_inputs: bool,
    fill_value: u8,
}

fn parse_mock_inference(node: &NodeSpec) -> Result<MockInferenceConfig> {
    let params = params_object(node)?;
    reject_unknown_fields(params, MOCK_INFERENCE_PARAM_FIELDS)?;
    let shape = read_shape(params, "shape", &[1, 4])?;
    let output_shape = read_shape(params, "output_shape", shape.dims())?;
    let dtype = read_dtype(params, "dtype")?.unwrap_or(DataType::F32);
    let output_dtype = read_dtype(params, "output_dtype")?.unwrap_or(dtype);
    let echo_inputs = read_bool(params, "echo_inputs", true)?;
    let fill_value = read_u8(params, "fill_value", 0)?;

    let input_info = TensorInfo::new(shape.clone(), dtype).with_layout(DataFormat::NC);
    let output_info = TensorInfo::new(output_shape, output_dtype).with_layout(DataFormat::NC);
    let option = RuntimeOption::new(
        BackendKind::Mock,
        ModelSource::Bytes(Vec::new()),
        BackendOptions::Mock(MockOptions {
            input_infos: vec![input_info],
            output_infos: vec![output_info],
            echo_inputs,
            fill_value,
        }),
    );
    Ok(MockInferenceConfig {
        option,
        echo_inputs,
        fill_value,
    })
}

fn create_sink(node: &NodeSpec) -> Result<CreatedElement> {
    validate_empty_params(node)?;
    let collector = Arc::new(Mutex::new(SinkCollector::default()));
    Ok(CreatedElement {
        element: Box::new(SinkElement {
            collector: collector.clone(),
        }),
        handle: ElementHandle::Sink(collector),
    })
}

fn validate_empty_params(node: &NodeSpec) -> Result<()> {
    if node.params.is_null() {
        return Ok(());
    }
    let params = params_object(node)?;
    reject_unknown_fields(params, &[])
}

fn params_object(node: &NodeSpec) -> Result<&Map<String, Value>> {
    node.params
        .as_object()
        .ok_or_else(|| Error::Config(format!("node {} params must be an object", node.name)))
}

fn reject_unknown_fields(params: &Map<String, Value>, allowed: &[&str]) -> Result<()> {
    for key in params.keys() {
        if !allowed.contains(&key.as_str()) {
            let message = if allowed.is_empty() {
                format!("unknown field `{key}`; no parameters are supported")
            } else {
                format!(
                    "unknown field `{key}`; expected one of {}",
                    allowed.join(", ")
                )
            };
            return Err(Error::Config(message));
        }
    }
    Ok(())
}

fn read_usize(params: &Map<String, Value>, key: &str, default: usize) -> Result<usize> {
    match params.get(key) {
        Some(value) => value
            .as_u64()
            .ok_or_else(|| Error::Config(format!("field {key} must be a non-negative integer")))
            .and_then(|value| {
                usize::try_from(value).map_err(|_| Error::Config(format!("field {key} overflow")))
            }),
        None => Ok(default),
    }
}

fn read_u8(params: &Map<String, Value>, key: &str, default: u8) -> Result<u8> {
    match params.get(key) {
        Some(value) => value
            .as_u64()
            .ok_or_else(|| Error::Config(format!("field {key} must be an integer")))
            .and_then(|value| {
                u8::try_from(value).map_err(|_| Error::Config(format!("field {key} overflow")))
            }),
        None => Ok(default),
    }
}

fn read_f32(params: &Map<String, Value>, key: &str, default: f32) -> Result<f32> {
    match params.get(key) {
        Some(value) => value
            .as_f64()
            .ok_or_else(|| Error::Config(format!("field {key} must be a number")))
            .and_then(|value| f64_to_exact_f32(value, key)),
        None => Ok(default),
    }
}

fn read_bool(params: &Map<String, Value>, key: &str, default: bool) -> Result<bool> {
    match params.get(key) {
        Some(value) => value
            .as_bool()
            .ok_or_else(|| Error::Config(format!("field {key} must be a boolean"))),
        None => Ok(default),
    }
}

fn read_shape(params: &Map<String, Value>, key: &str, default: &[usize]) -> Result<Shape> {
    match params.get(key) {
        Some(value) => {
            let array = value
                .as_array()
                .ok_or_else(|| Error::Config(format!("field {key} must be an array")))?;
            let dims = array
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .ok_or_else(|| Error::Config(format!("field {key} must contain integers")))
                        .and_then(|v| {
                            usize::try_from(v)
                                .map_err(|_| Error::Config(format!("field {key} overflow")))
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Shape::new(dims))
        }
        None => Ok(Shape::new(default.to_vec())),
    }
}

fn read_dtype(params: &Map<String, Value>, key: &str) -> Result<Option<DataType>> {
    let Some(value) = params.get(key) else {
        return Ok(None);
    };
    let name = value
        .as_str()
        .ok_or_else(|| Error::Config(format!("field {key} must be a string")))?;
    let dtype = match name {
        "f32" => DataType::F32,
        "f16" => DataType::F16,
        "bf16" => DataType::BF16,
        "u8" => DataType::U8,
        "i8" => DataType::I8,
        "u16" => DataType::U16,
        "i16" => DataType::I16,
        _ => return Err(Error::Config(format!("unsupported dtype: {name}"))),
    };
    Ok(Some(dtype))
}

fn read_format(params: &Map<String, Value>, key: &str) -> Result<Option<DataFormat>> {
    let Some(value) = params.get(key) else {
        return Ok(None);
    };
    let name = value
        .as_str()
        .ok_or_else(|| Error::Config(format!("field {key} must be a string")))?;
    let format = match name {
        "auto" => DataFormat::Auto,
        "nchw" => DataFormat::NCHW,
        "nhwc" => DataFormat::NHWC,
        "nc" => DataFormat::NC,
        "n" => DataFormat::N,
        "nc4hw" => DataFormat::NC4HW,
        "nc8hw" => DataFormat::NC8HW,
        "ncdhw" => DataFormat::NCDHW,
        "oihw" => DataFormat::OIHW,
        _ => return Err(Error::Config(format!("unsupported format: {name}"))),
    };
    Ok(Some(format))
}

fn filled_tensor(shape: Shape, dtype: DataType, format: DataFormat, value: f32) -> Result<Tensor> {
    let device = dg_core::CpuDevice::new();
    let desc = TensorDesc::new(shape.clone(), dtype, format, DeviceKind::Cpu);
    let tensor = Tensor::allocate(&device, desc)?;
    if dtype == DataType::F32 {
        let count = shape.element_count()?;
        let mut bytes = Vec::with_capacity(count * std::mem::size_of::<f32>());
        for _ in 0..count {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        tensor.buffer().write_from_slice(&bytes)?;
    } else if dtype == DataType::U8 {
        let count = shape.element_count()?;
        let byte = f32_to_exact_u8(value)?;
        tensor.buffer().write_from_slice(&vec![byte; count])?;
    } else {
        return Err(Error::Config(format!(
            "source element only supports f32/u8 for now, got {dtype:?}"
        )));
    }
    Ok(tensor)
}

fn usize_to_exact_f32(value: usize, field: &str) -> Result<f32> {
    const MAX_EXACT_INT: usize = 16_777_216;
    if value > MAX_EXACT_INT {
        return Err(Error::Config(format!(
            "{field} {value} cannot be represented exactly as f32"
        )));
    }
    let narrowed = value as f32;
    Ok(narrowed)
}

fn f64_to_exact_f32(value: f64, field: &str) -> Result<f32> {
    if !value.is_finite() {
        return Err(Error::Config(format!("field {field} must be finite")));
    }
    let narrowed = value as f32;
    if f64::from(narrowed) != value {
        return Err(Error::Config(format!(
            "field {field} {value} cannot be represented exactly as f32"
        )));
    }
    Ok(narrowed)
}

fn f32_to_exact_u8(value: f32) -> Result<u8> {
    if !value.is_finite() {
        return Err(Error::Config(
            "source element value must be finite to convert to u8".to_string(),
        ));
    }
    if !(0.0..=255.0).contains(&value) {
        return Err(Error::Config(format!(
            "source element value {value} cannot be represented as u8"
        )));
    }
    if value.fract() != 0.0 {
        return Err(Error::Config(format!(
            "source element value {value} cannot be represented as u8"
        )));
    }
    let text = value.to_string();
    text.parse::<u8>().map_err(|_| {
        Error::Config(format!(
            "source element value {value} cannot be represented as u8"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphFormat, GraphSpec};
    use serde_json::json;

    #[test]
    fn exact_f32_conversion_rejects_large_usize() {
        let err = usize_to_exact_f32(16_777_217, "index").expect_err("expected rejection");
        assert!(
            matches!(err, Error::Config(message) if message.contains("cannot be represented exactly as f32"))
        );
    }

    #[test]
    fn read_f32_rejects_precision_loss() {
        let mut params = Map::new();
        params.insert("start".to_string(), json!(0.1_f64 + f64::EPSILON));
        let err = read_f32(&params, "start", 0.0).expect_err("expected rejection");
        assert!(
            matches!(err, Error::Config(message) if message.contains("cannot be represented exactly as f32"))
        );
    }

    #[test]
    fn filled_tensor_rejects_u8_out_of_range_and_fractional_values() {
        let shape = Shape::new(vec![1]);
        let out_of_range = filled_tensor(shape.clone(), DataType::U8, DataFormat::NC, 256.0);
        assert!(
            matches!(out_of_range, Err(Error::Config(message)) if message.contains("cannot be represented as u8"))
        );

        let fractional = filled_tensor(shape, DataType::U8, DataFormat::NC, 1.5);
        assert!(
            matches!(fractional, Err(Error::Config(message)) if message.contains("cannot be represented as u8"))
        );
    }

    #[test]
    fn source_params_reject_unknown_fields_during_graph_load() {
        let yaml = r#"
apiVersion: dg/v1
kind: Graph
nodes:
  - name: source
    kind: source
    params:
      unexpected: true
"#;
        let err = GraphSpec::from_str_with_format(yaml, GraphFormat::Yaml)
            .expect("parse")
            .normalize_with_base_dir(None)
            .expect_err("unknown source parameter is rejected");
        let message = err.to_string();
        assert!(message.contains("nodes[source].params"));
        assert!(message.contains("unknown field `unexpected`"));
    }

    #[test]
    fn source_params_reject_invalid_enum_values_during_graph_load() {
        let yaml = r#"
apiVersion: dg/v1
kind: Graph
nodes:
  - name: source
    kind: source
    params:
      dtype: float32
"#;
        let err = GraphSpec::from_str_with_format(yaml, GraphFormat::Yaml)
            .expect("parse")
            .normalize_with_base_dir(None)
            .expect_err("unsupported dtype is rejected");
        let message = err.to_string();
        assert!(message.contains("nodes[source].params"));
        assert!(message.contains("unsupported dtype: float32"));
    }

    #[test]
    fn parameterless_elements_reject_params_during_graph_load() {
        let yaml = r#"
apiVersion: dg/v1
kind: Graph
nodes:
  - name: input
    kind: input
    params:
      capacity: 4
"#;
        let err = GraphSpec::from_str_with_format(yaml, GraphFormat::Yaml)
            .expect("parse")
            .normalize_with_base_dir(None)
            .expect_err("input parameters are rejected");
        let message = err.to_string();
        assert!(message.contains("nodes[input].params"));
        assert!(message.contains("no parameters are supported"));
    }

    #[test]
    fn parameterless_elements_allow_omitted_params() {
        let yaml = r#"
apiVersion: dg/v1
kind: Graph
nodes:
  - name: input
    kind: input
  - name: sink
    kind: sink
connections:
  - input.out -> sink.in
"#;
        GraphSpec::from_str_with_format(yaml, GraphFormat::Yaml)
            .expect("parse")
            .normalize_with_base_dir(None)
            .expect("parameterless nodes allow omitted params");
    }

    #[test]
    fn mock_inference_params_reject_invalid_types_during_graph_load() {
        let yaml = r#"
apiVersion: dg/v1
kind: Graph
nodes:
  - name: infer
    kind: mock_inference
    params:
      echo_inputs: [true]
"#;
        let err = GraphSpec::from_str_with_format(yaml, GraphFormat::Yaml)
            .expect("parse")
            .normalize_with_base_dir(None)
            .expect_err("invalid boolean is rejected");
        let message = err.to_string();
        assert!(message.contains("nodes[infer].params"));
        assert!(message.contains("field echo_inputs must be a boolean"));
    }
}
