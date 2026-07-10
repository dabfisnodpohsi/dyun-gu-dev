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

const SOURCE_OUTPUT_PORT: PortSchema = PortSchema {
    name: "out",
    dtype: Some(DataType::F32),
};
const INPUT_OUTPUT_PORT: PortSchema = PortSchema {
    name: "out",
    dtype: Some(DataType::F32),
};
const INFER_INPUT_PORT: PortSchema = PortSchema {
    name: "in",
    dtype: Some(DataType::F32),
};
const INFER_OUTPUT_PORT: PortSchema = PortSchema {
    name: "out",
    dtype: Some(DataType::F32),
};
const SINK_INPUT_PORT: PortSchema = PortSchema {
    name: "in",
    dtype: None,
};

inventory::submit! {
    ElementDescriptor {
        kind: "input",
        input_ports: &[],
        output_ports: &[INPUT_OUTPUT_PORT],
        create: create_input,
    }
}

inventory::submit! {
    ElementDescriptor {
        kind: "source",
        input_ports: &[],
        output_ports: &[SOURCE_OUTPUT_PORT],
        create: create_source,
    }
}

inventory::submit! {
    ElementDescriptor {
        kind: "mock_inference",
        input_ports: &[INFER_INPUT_PORT],
        output_ports: &[INFER_OUTPUT_PORT],
        create: create_mock_inference,
    }
}

inventory::submit! {
    ElementDescriptor {
        kind: "sink",
        input_ports: &[SINK_INPUT_PORT],
        output_ports: &[],
        create: create_sink,
    }
}

struct SourceElement {
    count: usize,
    shape: Shape,
    dtype: DataType,
    start: f32,
}

struct InputElement {
    queue: Arc<Mutex<VecDeque<Tensor>>>,
}

impl Element for SourceElement {
    fn run(self: Box<Self>, io: ElementIo) -> Result<()> {
        trace!(node = %io.name, count = self.count, "running source element");
        for index in 0..self.count {
            if io.stop.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(Error::NotRunning);
            }
            let step = usize_to_exact_f32(index, "source index")?;
            let tensor = filled_tensor(self.shape.clone(), self.dtype, self.start + step)?;
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
                    if io.stop.load(std::sync::atomic::Ordering::Relaxed) {
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
                    if io.stop.load(std::sync::atomic::Ordering::Relaxed) {
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
            } else {
                return Err(Error::Runtime(
                    "expected tensor or detections payload".to_string(),
                ));
            }
        }
    }
}

fn create_source(node: &NodeSpec) -> Result<CreatedElement> {
    let params = params_object(node)?;
    let count = read_usize(params, "count", 1)?;
    let shape = read_shape(params, "shape", &[1, 4])?;
    let dtype = read_dtype(params, "dtype").unwrap_or(DataType::F32);
    let start = read_f32(params, "start", 0.0)?;
    Ok(CreatedElement {
        element: Box::new(SourceElement {
            count,
            shape,
            dtype,
            start,
        }),
        handle: ElementHandle::None,
    })
}

fn create_input(_node: &NodeSpec) -> Result<CreatedElement> {
    let queue = Arc::new(Mutex::new(VecDeque::new()));
    Ok(CreatedElement {
        element: Box::new(InputElement {
            queue: queue.clone(),
        }),
        handle: ElementHandle::Input(queue),
    })
}

fn create_mock_inference(node: &NodeSpec) -> Result<CreatedElement> {
    let params = params_object(node)?;
    let shape = read_shape(params, "shape", &[1, 4])?;
    let output_shape = read_shape(params, "output_shape", shape.dims())?;
    let dtype = read_dtype(params, "dtype").unwrap_or(DataType::F32);
    let output_dtype = read_dtype(params, "output_dtype").unwrap_or(dtype);
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
    let runtime = Runtime::new(option)?;
    Ok(CreatedElement {
        element: Box::new(MockInferenceElement {
            runtime,
            echo_inputs,
            fill_value,
        }),
        handle: ElementHandle::None,
    })
}

fn create_sink(node: &NodeSpec) -> Result<CreatedElement> {
    let _ = params_object(node)?;
    let collector = Arc::new(Mutex::new(SinkCollector::default()));
    Ok(CreatedElement {
        element: Box::new(SinkElement {
            collector: collector.clone(),
        }),
        handle: ElementHandle::Sink(collector),
    })
}

fn params_object(node: &NodeSpec) -> Result<&Map<String, Value>> {
    node.params
        .as_object()
        .ok_or_else(|| Error::Config(format!("node {} params must be an object", node.name)))
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

fn read_dtype(params: &Map<String, Value>, key: &str) -> Option<DataType> {
    params
        .get(key)
        .and_then(|value| value.as_str())
        .and_then(|name| match name {
            "f32" => Some(DataType::F32),
            "f16" => Some(DataType::F16),
            "bf16" => Some(DataType::BF16),
            "u8" => Some(DataType::U8),
            "i8" => Some(DataType::I8),
            "u16" => Some(DataType::U16),
            "i16" => Some(DataType::I16),
            _ => None,
        })
}

fn filled_tensor(shape: Shape, dtype: DataType, value: f32) -> Result<Tensor> {
    let device = dg_core::CpuDevice::new();
    let desc = TensorDesc::new(shape.clone(), dtype, DataFormat::NC, DeviceKind::Cpu);
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
        let out_of_range = filled_tensor(shape.clone(), DataType::U8, 256.0);
        assert!(
            matches!(out_of_range, Err(Error::Config(message)) if message.contains("cannot be represented as u8"))
        );

        let fractional = filled_tensor(shape, DataType::U8, 1.5);
        assert!(
            matches!(fractional, Err(Error::Config(message)) if message.contains("cannot be represented as u8"))
        );
    }
}
