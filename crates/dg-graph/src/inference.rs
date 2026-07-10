use std::path::PathBuf;

use dg_core::{DataType, DeployMode, DeviceKind, Shape, TypeCode};
use dg_runtime::{configure_backend, BackendConfig, Runtime};
use serde::Deserialize;
use serde_json::Value;
use tracing::trace;

use crate::{
    CreatedElement, Element, ElementDescriptor, ElementHandle, ElementIo, Error, NodeSpec,
    PortSchema, Result,
};

const INPUT_PORT: PortSchema = PortSchema {
    name: "in",
    dtype: None,
};
const OUTPUT_PORT: PortSchema = PortSchema {
    name: "out",
    dtype: None,
};

inventory::submit! {
    ElementDescriptor {
        kind: "inference",
        input_ports: &[INPUT_PORT],
        output_ports: &[OUTPUT_PORT],
        create: create_inference,
    }
}

struct InferenceElement {
    runtime: Runtime,
}

impl Element for InferenceElement {
    fn run(mut self: Box<Self>, io: ElementIo) -> Result<()> {
        trace!(node = %io.name, backend = ?self.runtime.backend_kind(), "running inference element");
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

            let input = packet
                .tensor_ref()
                .ok_or_else(|| Error::Runtime("inference expects a tensor payload".to_string()))?
                .clone();
            let meta = packet.meta.clone();
            for output in self.runtime.run(&[input])? {
                io.send("out", crate::Packet::tensor(output).with_meta(meta.clone()))?;
            }
        }
    }
}

fn create_inference(node: &NodeSpec) -> Result<CreatedElement> {
    create_inference_inner(node.params.clone()).map_err(|err| match err {
        Error::Config(message) => {
            Error::Config(format!("node {} inference params: {message}", node.name))
        }
        err => Error::Element {
            element: node.name.clone(),
            message: err.to_string(),
        },
    })
}

fn create_inference_inner(value: Value) -> Result<CreatedElement> {
    let params: InferenceParams = serde_json::from_value(value)
        .map_err(|err| Error::Config(format!("invalid parameters: {err}")))?;
    let mut config = BackendConfig::new(params.model, params.options);
    if let Some(precision) = params.precision.as_deref() {
        config = config.with_precision(parse_dtype(precision)?);
    }
    if let Some(device) = params.device.as_deref() {
        config = config.with_device(parse_device(device)?);
    }
    if let Some(deploy_mode) = params.deploy_mode.as_deref() {
        config = config.with_deploy_mode(parse_deploy_mode(deploy_mode)?);
    }
    if let Some(core_mask) = params.core_mask {
        config = config.with_core_mask(core_mask);
    }

    let option = configure_backend(&params.backend, config)?;
    let mut runtime = Runtime::new(option)?;
    if runtime.input_count() != 1 {
        return Err(Error::Config(format!(
            "inference element requires a single-input model, got {} inputs",
            runtime.input_count()
        )));
    }
    if runtime.output_count() == 0 {
        return Err(Error::Config("inference model has no outputs".to_string()));
    }
    if let Some(shape) = params.reshape {
        runtime.reshape(&[Shape::new(shape)])?;
    }

    Ok(CreatedElement {
        element: Box::new(InferenceElement { runtime }),
        handle: ElementHandle::None,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InferenceParams {
    backend: String,
    #[serde(default)]
    model: Option<PathBuf>,
    #[serde(default)]
    precision: Option<String>,
    #[serde(default)]
    device: Option<String>,
    #[serde(default)]
    deploy_mode: Option<String>,
    #[serde(default)]
    core_mask: Option<u32>,
    #[serde(default)]
    reshape: Option<Vec<usize>>,
    #[serde(default)]
    options: Value,
}

fn parse_dtype(value: &str) -> Result<DataType> {
    match value {
        "f4" => Ok(DataType::F4),
        "f8" => Ok(DataType::F8),
        "f16" => Ok(DataType::F16),
        "f32" => Ok(DataType::F32),
        "f64" => Ok(DataType::F64),
        "bf16" => Ok(DataType::BF16),
        "u8" => Ok(DataType::U8),
        "u16" => Ok(DataType::U16),
        "u32" => Ok(DataType::new(TypeCode::Uint, 32, 1)),
        "u64" => Ok(DataType::new(TypeCode::Uint, 64, 1)),
        "i4" => Ok(DataType::I4),
        "i8" => Ok(DataType::I8),
        "i16" => Ok(DataType::I16),
        "i32" => Ok(DataType::new(TypeCode::Int, 32, 1)),
        "i64" => Ok(DataType::new(TypeCode::Int, 64, 1)),
        _ => Err(Error::Config(format!(
            "unsupported inference precision: {value}"
        ))),
    }
}

fn parse_device(value: &str) -> Result<DeviceKind> {
    match value {
        "cpu" => Ok(DeviceKind::Cpu),
        "intel_gpu" => Ok(DeviceKind::IntelGpu),
        "intel_npu" => Ok(DeviceKind::IntelNpu),
        "cuda" | "cuda_gpu" => Ok(DeviceKind::CudaGpu),
        "rknn" | "rknn_npu" => Ok(DeviceKind::RknnNpu),
        "sophon" | "sophon_tpu" => Ok(DeviceKind::SophonTpu),
        _ => Err(Error::Config(format!(
            "unsupported inference device: {value}"
        ))),
    }
}

fn parse_deploy_mode(value: &str) -> Result<DeployMode> {
    match value {
        "host" => Ok(DeployMode::Host),
        "soc" => Ok(DeployMode::SoC),
        _ => Err(Error::Config(format!(
            "unsupported inference deploy_mode: {value}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use crate::{Graph, GraphFormat, GraphSpec};

    #[test]
    fn generic_mock_inference_runs_in_graph() {
        let yaml = r#"
apiVersion: dg/v1
kind: Graph
nodes:
  - name: source
    kind: source
    params:
      count: 2
      shape: [1, 2]
  - name: infer
    kind: inference
    params:
      backend: mock
      reshape: [1, 2]
      options:
        shape: [1, 2]
        echo_inputs: true
  - name: sink
    kind: sink
    params: {}
connections:
  - source.out -> infer.in
  - infer.out -> sink.in
"#;
        let spec = GraphSpec::from_str_with_format(yaml, GraphFormat::Yaml)
            .expect("parse")
            .normalize_with_base_dir(None)
            .expect("normalize");
        let report = Graph::new(spec).expect("build").run().expect("run");
        let outputs = report.sinks.get("sink").expect("sink outputs");
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].desc().shape().dims(), &[1, 2]);
    }

    #[test]
    fn unknown_backend_is_rejected_with_node_context() {
        let yaml = r#"
apiVersion: dg/v1
kind: Graph
nodes:
  - name: infer
    kind: inference
    params:
      backend: tensorrt
"#;
        let spec = GraphSpec::from_str_with_format(yaml, GraphFormat::Yaml)
            .expect("parse")
            .normalize_with_base_dir(None)
            .expect("normalize");
        let graph = Graph::new(spec).expect("build");
        let err = graph.run().expect_err("backend is not registered");
        let message = err.to_string();
        assert!(message.contains("element error in infer"));
        assert!(message.contains("unsupported backend: tensorrt"));
    }

    #[test]
    fn inference_options_reject_unknown_fields_with_node_context() {
        let yaml = r#"
apiVersion: dg/v1
kind: Graph
nodes:
  - name: infer
    kind: inference
    params:
      backend: mock
      options:
        unknown: true
"#;
        let spec = GraphSpec::from_str_with_format(yaml, GraphFormat::Yaml)
            .expect("parse")
            .normalize_with_base_dir(None)
            .expect("normalize");
        let graph = Graph::new(spec).expect("build");
        let err = graph.run().expect_err("unknown option is rejected");
        let message = err.to_string();
        assert!(message.contains("element error in infer"));
        assert!(message.contains("unknown field"));
    }
}
