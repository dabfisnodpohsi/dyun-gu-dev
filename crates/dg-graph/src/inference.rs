use std::path::PathBuf;
use std::sync::OnceLock;

use dg_core::{DataType, DeployMode, DeviceKind, Shape, TypeCode};
use dg_runtime::{
    configure_backend, validate_runtime_option, BackendConfig, CoreSelection, Runtime,
    RuntimeOption,
};
use dg_scheduler::{
    InstancePool, Lease, Placement, Request, Scheduler, SchedulingPolicy, Topology,
};
use serde::Deserialize;
use serde_json::Value;
use tracing::trace;

use crate::{
    CreatedElement, Element, ElementDescriptor, ElementHandle, ElementIo, Error, NodeSpec,
    ParamField, ParamType, PortSchema, Result,
};

const INPUT_PORT: PortSchema = PortSchema {
    name: "in",
    dtype: None,
    required: true,
};
const OUTPUT_PORT: PortSchema = PortSchema {
    name: "out",
    dtype: None,
    required: false,
};
const PRECISION_VALUES: &[&str] = &[
    "f4", "f8", "f16", "f32", "f64", "bf16", "u8", "i8", "i4", "u16", "i16", "u32", "i32", "u64",
    "i64",
];
const DEVICE_VALUES: &[&str] = &[
    "cpu",
    "intel_gpu",
    "intel_npu",
    "cuda",
    "cuda_gpu",
    "rknn",
    "rknn_npu",
    "sophon",
    "sophon_tpu",
];
const DEPLOY_MODE_VALUES: &[&str] = &["host", "soc"];
const SCHEDULE_VALUES: &[&str] = &["least_loaded", "round_robin"];
const INFERENCE_PARAMS: &[ParamField] = &[
    ParamField {
        name: "backend",
        ty: ParamType::Str,
        required: true,
    },
    ParamField {
        name: "model",
        ty: ParamType::Str,
        required: false,
    },
    ParamField {
        name: "precision",
        ty: ParamType::Enum(PRECISION_VALUES),
        required: false,
    },
    ParamField {
        name: "device",
        ty: ParamType::Enum(DEVICE_VALUES),
        required: false,
    },
    ParamField {
        name: "deploy_mode",
        ty: ParamType::Enum(DEPLOY_MODE_VALUES),
        required: false,
    },
    ParamField {
        name: "core_mask",
        ty: ParamType::Uint,
        required: false,
    },
    ParamField {
        name: "instances",
        ty: ParamType::Uint,
        required: false,
    },
    ParamField {
        name: "schedule",
        ty: ParamType::Enum(SCHEDULE_VALUES),
        required: false,
    },
    ParamField {
        name: "reshape",
        ty: ParamType::Array(&ParamType::Uint),
        required: false,
    },
    ParamField {
        name: "options",
        ty: ParamType::Object,
        required: false,
    },
];

inventory::submit! {
    ElementDescriptor {
        kind: "inference",
        input_ports: &[INPUT_PORT],
        output_ports: &[OUTPUT_PORT],
        params: INFERENCE_PARAMS,
        validate: Some(validate_inference),
        create: create_inference,
    }
}

enum InferenceExecution {
    Single {
        runtime: Runtime,
        _lease: Option<Lease>,
    },
    Pool {
        runtimes: Vec<Runtime>,
        pool: InstancePool,
        policy: SchedulingPolicy,
    },
}

struct InferenceElement {
    execution: InferenceExecution,
}

impl Element for InferenceElement {
    fn run(mut self: Box<Self>, io: ElementIo) -> Result<()> {
        trace!(node = %io.name, "running inference element");
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

            let input = packet
                .tensor_ref()
                .ok_or_else(|| Error::Runtime("inference expects a tensor payload".to_string()))?
                .clone();
            let meta = packet.meta.clone();
            let affinity_key = packet.meta.stream_id.as_deref();
            let outputs = match &mut self.execution {
                InferenceExecution::Single { runtime, .. } => runtime.run(&[input])?,
                InferenceExecution::Pool {
                    runtimes,
                    pool,
                    policy,
                } => {
                    let checkout = pool.checkout(*policy, affinity_key).map_err(|error| {
                        Error::Runtime(format!(
                            "inference pool checkout failed for node {}: {error}",
                            io.name
                        ))
                    })?;
                    runtimes[checkout.instance_index()].run(&[input])?
                }
            };
            for output in outputs {
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
    let plan = prepare_inference(value)?;
    let execution = if plan.instances <= 1 || plan.option.device.is_none() {
        let (option, lease) = acquire_inference_lease(plan.option)?;
        let mut runtime = Runtime::new(option)?;
        validate_runtime_shape(&runtime)?;
        if let Some(ref shape) = plan.reshape {
            runtime.reshape(std::slice::from_ref(shape))?;
        }
        InferenceExecution::Single {
            runtime,
            _lease: lease,
        }
    } else {
        create_inference_pool(plan)?
    };

    Ok(CreatedElement {
        element: Box::new(InferenceElement { execution }),
        handle: ElementHandle::None,
    })
}

fn validate_runtime_shape(runtime: &Runtime) -> Result<()> {
    if runtime.input_count() != 1 {
        return Err(Error::Config(format!(
            "inference element requires a single-input model, got {} inputs",
            runtime.input_count()
        )));
    }
    if runtime.output_count() == 0 {
        return Err(Error::Config("inference model has no outputs".to_string()));
    }
    Ok(())
}

fn create_inference_pool(plan: InferencePlan) -> Result<InferenceExecution> {
    let kind = plan
        .option
        .device
        .ok_or_else(|| Error::Config("inference pool requires a selected device".to_string()))?;
    let scheduler = inference_scheduler()?;
    let core_selection = requested_core_selection(&plan.option);
    let instance_count = usize::try_from(plan.instances)
        .map_err(|_| Error::Config("inference instances value is too large".to_string()))?;
    let pool = InstancePool::new(scheduler.clone(), kind, instance_count, core_selection)
        .map_err(|error| Error::Config(format!("failed to create inference pool: {error}")))?;
    let mut runtimes = Vec::with_capacity(pool.instance_count());
    for placement in pool.placements() {
        let option = option_for_placement(&plan.option, *placement, scheduler);
        let mut runtime = Runtime::new(option)?;
        if runtimes.is_empty() {
            validate_runtime_shape(&runtime)?;
        }
        if let Some(ref shape) = plan.reshape {
            runtime.reshape(std::slice::from_ref(shape))?;
        }
        runtimes.push(runtime);
    }
    Ok(InferenceExecution::Pool {
        runtimes,
        pool,
        policy: plan.policy,
    })
}

fn requested_core_selection(option: &RuntimeOption) -> CoreSelection {
    if option.core.is_explicit() {
        option.core
    } else if let Some(mask) = option.core_mask {
        CoreSelection::Mask(mask)
    } else {
        CoreSelection::Auto
    }
}

fn option_for_placement(
    base: &RuntimeOption,
    placement: Placement,
    scheduler: &Scheduler,
) -> RuntimeOption {
    let mut option = base.clone();
    option.device = Some(placement.kind);
    option.device_id = Some(placement.device_id);
    option.core = CoreSelection::Single(placement.core_id);
    option.core_mask = None;
    if option.deploy_mode.is_none() {
        option.deploy_mode = Some(scheduler.topology().deployment());
    }
    option
}

fn acquire_inference_lease(mut option: RuntimeOption) -> Result<(RuntimeOption, Option<Lease>)> {
    let Some(kind) = option.device else {
        return Ok((option, None));
    };

    let scheduler = inference_scheduler()?;
    let core_selection = if option.core.is_explicit() {
        option.core
    } else if let Some(mask) = option.core_mask {
        CoreSelection::Mask(mask)
    } else {
        CoreSelection::Auto
    };
    let request = if let Some(device_id) = option.device_id {
        Request::explicit(kind, device_id, core_selection)
    } else {
        Request::auto(kind).with_core_selection(core_selection)
    };
    let lease = scheduler.acquire(request).map_err(|error| {
        Error::Config(format!(
            "scheduler could not acquire {kind:?} device/core lease: {error}"
        ))
    })?;
    let (leased_kind, device_id) = lease.device();
    option.device = Some(leased_kind);
    option.device_id = Some(device_id);
    option.core = CoreSelection::Single(lease.core_id());
    option.core_mask = None;
    if option.deploy_mode.is_none() {
        option.deploy_mode = Some(scheduler.topology().deployment());
    }
    Ok((option, Some(lease)))
}

fn inference_scheduler() -> Result<&'static Scheduler> {
    static SCHEDULER: OnceLock<std::result::Result<Scheduler, String>> = OnceLock::new();
    match SCHEDULER.get_or_init(|| {
        Topology::from_registered_devices(1)
            .and_then(Scheduler::new)
            .map_err(|error| error.to_string())
    }) {
        Ok(scheduler) => Ok(scheduler),
        Err(error) => Err(Error::Config(format!(
            "failed to initialize inference scheduler: {error}"
        ))),
    }
}

#[cfg(test)]
fn inference_scheduler_snapshot() -> Result<Vec<dg_scheduler::DeviceLoad>> {
    inference_scheduler()?
        .snapshot()
        .map_err(|error| Error::Runtime(format!("failed to inspect inference scheduler: {error}")))
}

fn validate_inference(node: &NodeSpec) -> Result<()> {
    prepare_inference(node.params.clone()).map(|_| ())
}

struct InferencePlan {
    option: RuntimeOption,
    reshape: Option<Shape>,
    instances: u32,
    policy: SchedulingPolicy,
}

fn prepare_inference(value: Value) -> Result<InferencePlan> {
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
    validate_runtime_option(&option)?;
    Ok(InferencePlan {
        option,
        reshape: params.reshape.map(Shape::new),
        instances: params.instances,
        policy: parse_schedule(&params.schedule)?,
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
    #[serde(default = "default_instances")]
    instances: u32,
    #[serde(default)]
    schedule: Option<String>,
    #[serde(default)]
    reshape: Option<Vec<usize>>,
    #[serde(default)]
    options: Value,
}

fn default_instances() -> u32 {
    1
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

fn parse_schedule(value: &Option<String>) -> Result<SchedulingPolicy> {
    match value.as_deref().unwrap_or("least_loaded") {
        "least_loaded" => Ok(SchedulingPolicy::LeastLoaded),
        "round_robin" => Ok(SchedulingPolicy::RoundRobin),
        value => Err(Error::Config(format!(
            "unsupported inference schedule: {value}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::inference_scheduler_snapshot;
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
    fn device_selected_inference_acquires_and_releases_scheduler_lease() {
        let yaml = r#"
apiVersion: dg/v1
kind: Graph
nodes:
  - name: source
    kind: source
    params:
      count: 1
      shape: [1, 2]
  - name: infer
    kind: inference
    params:
      backend: mock
      device: cpu
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
        assert_eq!(
            report.sinks.get("sink").expect("sink outputs")[0]
                .buffer()
                .read_bytes(),
            vec![0, 0, 0, 0, 0, 0, 0, 0]
        );
        let snapshot = inference_scheduler_snapshot().expect("scheduler snapshot");
        assert!(snapshot
            .iter()
            .flat_map(|device| device.cores.iter())
            .all(|core| core.load == 0));
    }

    #[test]
    fn multi_instance_inference_dispatches_with_round_robin() {
        let yaml = r#"
apiVersion: dg/v1
kind: Graph
nodes:
  - name: source
    kind: source
    params:
      count: 4
      shape: [1, 2]
  - name: infer
    kind: inference
    params:
      backend: mock
      device: cpu
      instances: 3
      schedule: round_robin
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
        assert_eq!(report.sinks.get("sink").expect("sink outputs").len(), 4);
        let snapshot = inference_scheduler_snapshot().expect("scheduler snapshot");
        assert!(snapshot
            .iter()
            .flat_map(|device| device.cores.iter())
            .all(|core| core.load == 0));
    }

    #[test]
    fn multi_instance_inference_accepts_least_loaded_schedule() {
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
      device: cpu
      instances: 2
      schedule: least_loaded
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
        assert_eq!(report.sinks.get("sink").expect("sink outputs").len(), 2);
    }

    #[test]
    fn unknown_backend_is_rejected_during_graph_load() {
        let yaml = r#"
apiVersion: dg/v1
kind: Graph
nodes:
  - name: infer
    kind: inference
    params:
      backend: tensorrt
"#;
        let err = GraphSpec::from_str_with_format(yaml, GraphFormat::Yaml)
            .expect("parse")
            .normalize_with_base_dir(None)
            .expect_err("backend is not registered");
        let message = err.to_string();
        assert!(message.contains("nodes[infer].params"));
        assert!(message.contains("unsupported backend: tensorrt"));
    }

    #[test]
    fn inference_options_reject_unknown_fields_during_graph_load() {
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
        let err = GraphSpec::from_str_with_format(yaml, GraphFormat::Yaml)
            .expect("parse")
            .normalize_with_base_dir(None)
            .expect_err("unknown option is rejected");
        let message = err.to_string();
        assert!(message.contains("nodes[infer].params"));
        assert!(message.contains("unknown field"));
    }
}
