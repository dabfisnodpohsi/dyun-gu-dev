//! Stable C ABI for graph construction, tensor exchange, and external buffers.

#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::ffi::{c_char, c_int, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::ptr;

use dg_core::{
    Buffer, BufferDesc, CpuDevice, DataFormat, DataType, DeviceKind, ExternalDropGuard,
    ExternalHandle, MemoryDomain, Shape, Tensor, TensorDesc, TypeCode,
};
use dg_graph::{Graph, GraphDiff, GraphFormat, GraphSpec, NodeSpec};
use serde_json::{Map, Value};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
    static LAST_DATA: RefCell<Option<Box<[u8]>>> = const { RefCell::new(None) };
}

/// ABI status returned by every fallible C entry point.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DgStatus {
    Ok = 0,
    Again = 1,
    EndOfStream = 2,
    InvalidArgument = -1,
    NullPointer = -2,
    InvalidHandle = -3,
    ParseError = -4,
    NotBuilt = -5,
    RuntimeError = -6,
    Unsupported = -7,
    Panic = -8,
    InternalError = -9,
}

/// Graph serialization format.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DgGraphFormat {
    Yaml = 0,
    Json = 1,
    Toml = 2,
}

/// Supported tensor element types.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DgDataType {
    U8 = 0,
    U16 = 1,
    I4 = 2,
    I8 = 3,
    I16 = 4,
    F4 = 5,
    F8 = 6,
    F16 = 7,
    Bf16 = 8,
    F32 = 9,
    F64 = 10,
    U32 = 11,
    I32 = 12,
    U64 = 13,
    I64 = 14,
}

/// Tensor layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DgDataFormat {
    Auto = 0,
    N = 1,
    Nc = 2,
    Nchw = 3,
    Nhwc = 4,
    Nc4hw = 5,
    Nc8hw = 6,
    Ncdhw = 7,
    Oihw = 8,
}

/// Logical device family.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DgDeviceKind {
    Cpu = 0,
    IntelGpu = 1,
    IntelNpu = 2,
    CudaGpu = 3,
    RknnNpu = 4,
    SophonTpu = 5,
}

/// Imported external memory domain.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DgMemoryDomain {
    Host = 0,
    DmaBuf = 1,
    DrmPrime = 2,
    VaapiSurface = 3,
    CudaDevice = 4,
    MppBuffer = 5,
    SophonDevice = 6,
    Opaque = 7,
}

/// Opaque graph engine handle.
pub struct DgEngine {
    inner: Engine,
}

/// Opaque tensor handle.
pub struct DgTensor {
    tensor: Tensor,
}

/// Opaque buffer handle.
pub struct DgBuffer {
    buffer: Buffer,
}

struct Engine {
    spec: GraphSpec,
    graph: Option<Graph>,
    outputs: VecDeque<Tensor>,
    pending_inputs: BTreeMap<String, Vec<Tensor>>,
}

impl Engine {
    fn new() -> Self {
        Self {
            spec: GraphSpec::default(),
            graph: None,
            outputs: VecDeque::new(),
            pending_inputs: BTreeMap::new(),
        }
    }

    fn invalidate(&mut self) {
        self.graph = None;
        self.outputs.clear();
        self.pending_inputs.clear();
    }

    fn input_node_name(&self) -> Result<String, String> {
        let mut names = self
            .spec
            .nodes
            .iter()
            .filter(|node| node.kind == "input")
            .map(|node| node.name.clone());
        match (names.next(), names.next()) {
            (Some(name), None) => Ok(name),
            _ => Err("graph must contain exactly one input node".to_string()),
        }
    }

    fn push(&mut self, tensor: Tensor) -> Result<(), String> {
        if self.graph.is_none() {
            return Err("engine must be built before pushing input".to_string());
        }
        let input_name = self.input_node_name()?;
        self.pending_inputs
            .entry(input_name)
            .or_default()
            .push(tensor);
        Ok(())
    }

    fn run(&mut self) -> Result<(), String> {
        let graph = self
            .graph
            .as_ref()
            .ok_or_else(|| "engine must be built first".to_string())?;
        let inputs = std::mem::take(&mut self.pending_inputs)
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>();
        let report = graph
            .run_with_inputs(inputs)
            .map_err(|error| error.to_string())?;
        for tensors in report.sinks.into_values() {
            self.outputs.extend(tensors);
        }
        Ok(())
    }
}

fn set_error(message: impl Into<String>) {
    let message = message.into().replace('\0', " ");
    let value = CString::new(message)
        .unwrap_or_else(|_| CString::new("unknown error").expect("literal has no NUL"));
    LAST_ERROR.with(|last| *last.borrow_mut() = Some(value));
}

fn clear_error() {
    LAST_ERROR.with(|last| *last.borrow_mut() = None);
}

fn ffi_result<T>(operation: impl FnOnce() -> Result<T, (DgStatus, String)>) -> Result<T, DgStatus> {
    clear_error();
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err((status, message))) => {
            set_error(message);
            Err(status)
        }
        Err(_) => {
            set_error("panic crossed C ABI boundary");
            Err(DgStatus::Panic)
        }
    }
}

fn graph_error(error: impl std::fmt::Display) -> (DgStatus, String) {
    (DgStatus::ParseError, error.to_string())
}

fn write_diff_counts(
    diff: &GraphDiff,
    out_added_nodes: *mut usize,
    out_removed_nodes: *mut usize,
    out_updated_nodes: *mut usize,
    out_added_connections: *mut usize,
    out_removed_connections: *mut usize,
) -> Result<(), (DgStatus, String)> {
    validate_diff_outputs(
        out_added_nodes,
        out_removed_nodes,
        out_updated_nodes,
        out_added_connections,
        out_removed_connections,
    )?;
    // SAFETY: all output pointers were checked non-null by `validate_diff_outputs`.
    unsafe {
        out_added_nodes.write(diff.added_nodes.len());
        out_removed_nodes.write(diff.removed_nodes.len());
        out_updated_nodes.write(diff.updated_nodes.len());
        out_added_connections.write(diff.added_connections.len());
        out_removed_connections.write(diff.removed_connections.len());
    }
    Ok(())
}

fn validate_diff_outputs(
    out_added_nodes: *mut usize,
    out_removed_nodes: *mut usize,
    out_updated_nodes: *mut usize,
    out_added_connections: *mut usize,
    out_removed_connections: *mut usize,
) -> Result<(), (DgStatus, String)> {
    if out_added_nodes.is_null()
        || out_removed_nodes.is_null()
        || out_updated_nodes.is_null()
        || out_added_connections.is_null()
        || out_removed_connections.is_null()
    {
        return Err((
            DgStatus::NullPointer,
            "diff output pointer is null".to_string(),
        ));
    }
    Ok(())
}

fn format_from_c(format: DgGraphFormat) -> GraphFormat {
    match format {
        DgGraphFormat::Yaml => GraphFormat::Yaml,
        DgGraphFormat::Json => GraphFormat::Json,
        DgGraphFormat::Toml => GraphFormat::Toml,
    }
}

fn data_type_from_c(dtype: DgDataType) -> DataType {
    match dtype {
        DgDataType::U8 => DataType::U8,
        DgDataType::U16 => DataType::U16,
        DgDataType::I4 => DataType::I4,
        DgDataType::I8 => DataType::I8,
        DgDataType::I16 => DataType::I16,
        DgDataType::F4 => DataType::F4,
        DgDataType::F8 => DataType::F8,
        DgDataType::F16 => DataType::F16,
        DgDataType::Bf16 => DataType::BF16,
        DgDataType::F32 => DataType::F32,
        DgDataType::F64 => DataType::F64,
        DgDataType::U32 => DataType::new(TypeCode::Uint, 32, 1),
        DgDataType::I32 => DataType::new(TypeCode::Int, 32, 1),
        DgDataType::U64 => DataType::new(TypeCode::Uint, 64, 1),
        DgDataType::I64 => DataType::new(TypeCode::Int, 64, 1),
    }
}

fn format_from_c_enum(format: DgDataFormat) -> DataFormat {
    match format {
        DgDataFormat::Auto => DataFormat::Auto,
        DgDataFormat::N => DataFormat::N,
        DgDataFormat::Nc => DataFormat::NC,
        DgDataFormat::Nchw => DataFormat::NCHW,
        DgDataFormat::Nhwc => DataFormat::NHWC,
        DgDataFormat::Nc4hw => DataFormat::NC4HW,
        DgDataFormat::Nc8hw => DataFormat::NC8HW,
        DgDataFormat::Ncdhw => DataFormat::NCDHW,
        DgDataFormat::Oihw => DataFormat::OIHW,
    }
}

fn device_from_c(device: DgDeviceKind) -> DeviceKind {
    match device {
        DgDeviceKind::Cpu => DeviceKind::Cpu,
        DgDeviceKind::IntelGpu => DeviceKind::IntelGpu,
        DgDeviceKind::IntelNpu => DeviceKind::IntelNpu,
        DgDeviceKind::CudaGpu => DeviceKind::CudaGpu,
        DgDeviceKind::RknnNpu => DeviceKind::RknnNpu,
        DgDeviceKind::SophonTpu => DeviceKind::SophonTpu,
    }
}

fn domain_from_c(domain: DgMemoryDomain) -> MemoryDomain {
    match domain {
        DgMemoryDomain::Host => MemoryDomain::Host,
        DgMemoryDomain::DmaBuf => MemoryDomain::DmaBuf,
        DgMemoryDomain::DrmPrime => MemoryDomain::DrmPrime,
        DgMemoryDomain::VaapiSurface => MemoryDomain::VaapiSurface,
        DgMemoryDomain::CudaDevice => MemoryDomain::CudaDevice,
        DgMemoryDomain::MppBuffer => MemoryDomain::MppBuffer,
        DgMemoryDomain::SophonDevice => MemoryDomain::SophonDevice,
        DgMemoryDomain::Opaque => MemoryDomain::Opaque,
    }
}

unsafe fn bytes<'a>(data: *const u8, length: usize) -> Result<&'a [u8], (DgStatus, String)> {
    if length == 0 {
        return Ok(&[]);
    }
    if data.is_null() {
        return Err((DgStatus::NullPointer, "data pointer is null".to_string()));
    }
    // SAFETY: the caller must provide a readable region of `length` bytes.
    Ok(unsafe { std::slice::from_raw_parts(data, length) })
}

unsafe fn dims<'a>(values: *const usize, rank: usize) -> Result<&'a [usize], (DgStatus, String)> {
    if rank == 0 {
        return Ok(&[]);
    }
    if values.is_null() {
        return Err((DgStatus::NullPointer, "shape pointer is null".to_string()));
    }
    // SAFETY: the caller must provide `rank` readable shape dimensions.
    Ok(unsafe { std::slice::from_raw_parts(values, rank) })
}

unsafe fn c_string<'a>(value: *const c_char) -> Result<&'a CStr, (DgStatus, String)> {
    if value.is_null() {
        return Err((DgStatus::NullPointer, "string pointer is null".to_string()));
    }
    // SAFETY: the caller must provide a valid NUL-terminated C string.
    Ok(unsafe { CStr::from_ptr(value) })
}

fn tensor_from_bytes(
    data: &[u8],
    shape: &[usize],
    dtype: DgDataType,
    format: DgDataFormat,
    device: DgDeviceKind,
) -> Result<Tensor, (DgStatus, String)> {
    let desc = TensorDesc::new(
        Shape::new(shape.to_vec()),
        data_type_from_c(dtype),
        format_from_c_enum(format),
        device_from_c(device),
    );
    let expected = desc
        .storage_bytes()
        .map_err(|error| (DgStatus::InvalidArgument, error.to_string()))?;
    if expected != data.len() {
        return Err((
            DgStatus::InvalidArgument,
            format!(
                "tensor byte length {}/{} does not match shape and dtype",
                data.len(),
                expected
            ),
        ));
    }
    let tensor = Tensor::allocate(&CpuDevice::new(), desc)
        .map_err(|error| (DgStatus::RuntimeError, error.to_string()))?;
    tensor
        .buffer()
        .write_from_slice(data)
        .map_err(|error| (DgStatus::RuntimeError, error.to_string()))?;
    Ok(tensor)
}

/// Returns the most recent error for the calling thread.
#[no_mangle]
pub extern "C" fn dg_last_error() -> *const c_char {
    LAST_ERROR.with(|last| {
        last.borrow()
            .as_ref()
            .map_or(ptr::null(), |message| message.as_ptr())
    })
}

/// Returns the package version as a static UTF-8 C string.
#[no_mangle]
pub extern "C" fn dg_version() -> *const c_char {
    c"0.1.0".as_ptr()
}

/// Creates an engine handle.
#[no_mangle]
pub unsafe extern "C" fn dg_engine_create(out: *mut *mut DgEngine) -> DgStatus {
    match ffi_result(|| {
        if out.is_null() {
            return Err((
                DgStatus::NullPointer,
                "engine output pointer is null".to_string(),
            ));
        }
        let handle = Box::new(DgEngine {
            inner: Engine::new(),
        });
        // SAFETY: `out` was checked non-null and points to writable caller storage.
        unsafe { out.write(Box::into_raw(handle)) };
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Frees an engine handle. Null is accepted.
#[no_mangle]
pub unsafe extern "C" fn dg_engine_free(engine: *mut DgEngine) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if !engine.is_null() {
            // SAFETY: the pointer must have been returned by `dg_engine_create` exactly once.
            unsafe { drop(Box::from_raw(engine)) };
        }
    }));
}

/// Loads a graph specification from a UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn dg_engine_load_string(
    engine: *mut DgEngine,
    format: DgGraphFormat,
    content: *const c_char,
) -> DgStatus {
    match ffi_result(|| {
        if engine.is_null() {
            return Err((DgStatus::NullPointer, "engine pointer is null".to_string()));
        }
        let content = unsafe { c_string(content)? }
            .to_str()
            .map_err(|error| (DgStatus::InvalidArgument, error.to_string()))?;
        let spec =
            GraphSpec::from_str_with_format(content, format_from_c(format)).map_err(graph_error)?;
        spec.validate().map_err(graph_error)?;
        let engine = unsafe { &mut *engine };
        engine.inner.spec = spec;
        engine.inner.invalidate();
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Loads a graph specification from a UTF-8 path.
#[no_mangle]
pub unsafe extern "C" fn dg_engine_load_file(
    engine: *mut DgEngine,
    path: *const c_char,
) -> DgStatus {
    match ffi_result(|| {
        if engine.is_null() {
            return Err((DgStatus::NullPointer, "engine pointer is null".to_string()));
        }
        let path = unsafe { c_string(path)? }
            .to_str()
            .map_err(|error| (DgStatus::InvalidArgument, error.to_string()))?;
        let spec = GraphSpec::load_from_path(Path::new(path)).map_err(graph_error)?;
        spec.validate().map_err(graph_error)?;
        let engine = unsafe { &mut *engine };
        engine.inner.spec = spec;
        engine.inner.invalidate();
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Reloads a graph specification from a UTF-8 string and invalidates the built graph.
#[no_mangle]
pub unsafe extern "C" fn dg_engine_reload_string(
    engine: *mut DgEngine,
    format: DgGraphFormat,
    content: *const c_char,
) -> DgStatus {
    match ffi_result(|| {
        if engine.is_null() {
            return Err((DgStatus::NullPointer, "engine pointer is null".to_string()));
        }
        let content = unsafe { c_string(content)? }
            .to_str()
            .map_err(|error| (DgStatus::InvalidArgument, error.to_string()))?;
        let spec =
            GraphSpec::from_str_with_format(content, format_from_c(format)).map_err(graph_error)?;
        spec.validate().map_err(graph_error)?;
        let engine = unsafe { &mut *engine };
        engine.inner.spec = spec;
        engine.inner.invalidate();
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Reloads a graph specification from a UTF-8 path and invalidates the built graph.
#[no_mangle]
pub unsafe extern "C" fn dg_engine_reload_file(
    engine: *mut DgEngine,
    path: *const c_char,
) -> DgStatus {
    match ffi_result(|| {
        if engine.is_null() {
            return Err((DgStatus::NullPointer, "engine pointer is null".to_string()));
        }
        let path = unsafe { c_string(path)? }
            .to_str()
            .map_err(|error| (DgStatus::InvalidArgument, error.to_string()))?;
        let spec = GraphSpec::load_from_path(Path::new(path)).map_err(graph_error)?;
        let engine = unsafe { &mut *engine };
        engine.inner.spec = spec;
        engine.inner.invalidate();
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Computes node and connection changes against a UTF-8 graph specification.
#[no_mangle]
pub unsafe extern "C" fn dg_engine_diff_string(
    engine: *const DgEngine,
    format: DgGraphFormat,
    content: *const c_char,
    out_added_nodes: *mut usize,
    out_removed_nodes: *mut usize,
    out_updated_nodes: *mut usize,
    out_added_connections: *mut usize,
    out_removed_connections: *mut usize,
) -> DgStatus {
    match ffi_result(|| {
        if engine.is_null() {
            return Err((DgStatus::NullPointer, "engine pointer is null".to_string()));
        }
        validate_diff_outputs(
            out_added_nodes,
            out_removed_nodes,
            out_updated_nodes,
            out_added_connections,
            out_removed_connections,
        )?;
        let content = unsafe { c_string(content)? }
            .to_str()
            .map_err(|error| (DgStatus::InvalidArgument, error.to_string()))?;
        let spec =
            GraphSpec::from_str_with_format(content, format_from_c(format)).map_err(graph_error)?;
        spec.validate().map_err(graph_error)?;
        let engine = unsafe { &*engine };
        let diff = Graph::diff(&engine.inner.spec, &spec);
        write_diff_counts(
            &diff,
            out_added_nodes,
            out_removed_nodes,
            out_updated_nodes,
            out_added_connections,
            out_removed_connections,
        )
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Computes node and connection changes against a UTF-8 graph file.
#[no_mangle]
pub unsafe extern "C" fn dg_engine_diff_file(
    engine: *const DgEngine,
    path: *const c_char,
    out_added_nodes: *mut usize,
    out_removed_nodes: *mut usize,
    out_updated_nodes: *mut usize,
    out_added_connections: *mut usize,
    out_removed_connections: *mut usize,
) -> DgStatus {
    match ffi_result(|| {
        if engine.is_null() {
            return Err((DgStatus::NullPointer, "engine pointer is null".to_string()));
        }
        validate_diff_outputs(
            out_added_nodes,
            out_removed_nodes,
            out_updated_nodes,
            out_added_connections,
            out_removed_connections,
        )?;
        let path = unsafe { c_string(path)? }
            .to_str()
            .map_err(|error| (DgStatus::InvalidArgument, error.to_string()))?;
        let spec = GraphSpec::load_from_path(Path::new(path)).map_err(graph_error)?;
        let engine = unsafe { &*engine };
        let diff = Graph::diff(&engine.inner.spec, &spec);
        write_diff_counts(
            &diff,
            out_added_nodes,
            out_removed_nodes,
            out_updated_nodes,
            out_added_connections,
            out_removed_connections,
        )
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Adds a node programmatically. `params_json` may be null for an empty object.
#[no_mangle]
pub unsafe extern "C" fn dg_engine_add_node(
    engine: *mut DgEngine,
    name: *const c_char,
    kind: *const c_char,
    params_json: *const c_char,
) -> DgStatus {
    match ffi_result(|| {
        if engine.is_null() {
            return Err((DgStatus::NullPointer, "engine pointer is null".to_string()));
        }
        let name = unsafe { c_string(name)? }
            .to_str()
            .map_err(|error| (DgStatus::InvalidArgument, error.to_string()))?
            .to_string();
        let kind = unsafe { c_string(kind)? }
            .to_str()
            .map_err(|error| (DgStatus::InvalidArgument, error.to_string()))?
            .to_string();
        let params = if params_json.is_null() {
            Value::Object(Map::new())
        } else {
            let params = unsafe { c_string(params_json)? }
                .to_str()
                .map_err(|error| (DgStatus::InvalidArgument, error.to_string()))?;
            serde_json::from_str(params)
                .map_err(|error| (DgStatus::ParseError, error.to_string()))?
        };
        let engine = unsafe { &mut *engine };
        engine.inner.spec.nodes.push(NodeSpec {
            name,
            kind,
            template: None,
            params,
        });
        engine.inner.invalidate();
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Removes a node by name and its incident connections.
#[no_mangle]
pub unsafe extern "C" fn dg_engine_remove_node(
    engine: *mut DgEngine,
    name: *const c_char,
) -> DgStatus {
    match ffi_result(|| {
        if engine.is_null() {
            return Err((DgStatus::NullPointer, "engine pointer is null".to_string()));
        }
        let name = unsafe { c_string(name)? }
            .to_str()
            .map_err(|error| (DgStatus::InvalidArgument, error.to_string()))?;
        let engine = unsafe { &mut *engine };
        engine.inner.spec.nodes.retain(|node| node.name != name);
        engine.inner.spec.connections.retain(|connection| {
            dg_graph::ConnectionSpec::parse(connection)
                .is_ok_and(|parsed| parsed.from_node != name && parsed.to_node != name)
        });
        engine.inner.invalidate();
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Adds a graph edge in `source.port -> destination.port` form.
#[no_mangle]
pub unsafe extern "C" fn dg_engine_connect(
    engine: *mut DgEngine,
    connection: *const c_char,
) -> DgStatus {
    match ffi_result(|| {
        if engine.is_null() {
            return Err((DgStatus::NullPointer, "engine pointer is null".to_string()));
        }
        let connection = unsafe { c_string(connection)? }
            .to_str()
            .map_err(|error| (DgStatus::InvalidArgument, error.to_string()))?;
        dg_graph::ConnectionSpec::parse(connection).map_err(graph_error)?;
        let engine = unsafe { &mut *engine };
        engine.inner.spec.connections.push(connection.to_string());
        engine.inner.invalidate();
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Removes a graph edge.
#[no_mangle]
pub unsafe extern "C" fn dg_engine_disconnect(
    engine: *mut DgEngine,
    connection: *const c_char,
) -> DgStatus {
    match ffi_result(|| {
        if engine.is_null() {
            return Err((DgStatus::NullPointer, "engine pointer is null".to_string()));
        }
        let connection = unsafe { c_string(connection)? }
            .to_str()
            .map_err(|error| (DgStatus::InvalidArgument, error.to_string()))?;
        let engine = unsafe { &mut *engine };
        engine
            .inner
            .spec
            .connections
            .retain(|item| item != connection);
        engine.inner.invalidate();
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Validates and builds the current graph specification.
#[no_mangle]
pub unsafe extern "C" fn dg_engine_build(engine: *mut DgEngine) -> DgStatus {
    match ffi_result(|| {
        if engine.is_null() {
            return Err((DgStatus::NullPointer, "engine pointer is null".to_string()));
        }
        let engine = unsafe { &mut *engine };
        engine.inner.spec.validate().map_err(graph_error)?;
        engine.inner.graph = Some(Graph::new(engine.inner.spec.clone()).map_err(graph_error)?);
        engine.inner.outputs.clear();
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Runs the built graph with pending inputs and stores sink outputs for polling.
#[no_mangle]
pub unsafe extern "C" fn dg_engine_run(engine: *mut DgEngine) -> DgStatus {
    match ffi_result(|| {
        if engine.is_null() {
            return Err((DgStatus::NullPointer, "engine pointer is null".to_string()));
        }
        let engine = unsafe { &mut *engine };
        engine.inner.run().map_err(|message| {
            if message.contains("built") {
                (DgStatus::NotBuilt, message)
            } else {
                (DgStatus::RuntimeError, message)
            }
        })?;
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Creates a host tensor from a caller-owned byte array.
#[no_mangle]
pub unsafe extern "C" fn dg_tensor_create(
    data: *const u8,
    length: usize,
    shape: *const usize,
    rank: usize,
    dtype: DgDataType,
    format: DgDataFormat,
    device: DgDeviceKind,
    out: *mut *mut DgTensor,
) -> DgStatus {
    match ffi_result(|| {
        if out.is_null() {
            return Err((
                DgStatus::NullPointer,
                "tensor output pointer is null".to_string(),
            ));
        }
        let data = unsafe { bytes(data, length)? };
        let shape = unsafe { dims(shape, rank)? };
        let tensor = tensor_from_bytes(data, shape, dtype, format, device)?;
        // SAFETY: `out` was checked non-null and points to writable caller storage.
        unsafe { out.write(Box::into_raw(Box::new(DgTensor { tensor }))) };
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Creates a tensor backed by an imported external buffer.
#[no_mangle]
pub unsafe extern "C" fn dg_tensor_create_external(
    fd: c_int,
    raw: u64,
    domain: DgMemoryDomain,
    size_bytes: usize,
    shape: *const usize,
    rank: usize,
    dtype: DgDataType,
    format: DgDataFormat,
    device: DgDeviceKind,
    out: *mut *mut DgTensor,
) -> DgStatus {
    match ffi_result(|| {
        if out.is_null() {
            return Err((
                DgStatus::NullPointer,
                "tensor output pointer is null".to_string(),
            ));
        }
        let shape = unsafe { dims(shape, rank)? };
        let desc = TensorDesc::new(
            Shape::new(shape.to_vec()),
            data_type_from_c(dtype),
            format_from_c_enum(format),
            device_from_c(device),
        );
        let expected = desc
            .storage_bytes()
            .map_err(|error| (DgStatus::InvalidArgument, error.to_string()))?;
        if expected != size_bytes {
            return Err((
                DgStatus::InvalidArgument,
                format!("external size {size_bytes} does not match tensor size {expected}"),
            ));
        }
        let external = if fd >= 0 {
            ExternalHandle::from_fd(fd)
        } else {
            ExternalHandle::from_raw(raw)
        };
        let buffer = Buffer::from_external(
            device_from_c(device),
            domain_from_c(domain),
            BufferDesc::new(size_bytes, 1),
            external,
            vec![0; size_bytes],
            ExternalDropGuard::new(|| {}),
        )
        .map_err(|error| (DgStatus::RuntimeError, error.to_string()))?;
        let tensor = Tensor::from_buffer(desc, buffer)
            .map_err(|error| (DgStatus::RuntimeError, error.to_string()))?;
        // SAFETY: `out` was checked non-null and points to writable caller storage.
        unsafe { out.write(Box::into_raw(Box::new(DgTensor { tensor }))) };
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Frees a tensor handle. Null is accepted.
#[no_mangle]
pub unsafe extern "C" fn dg_tensor_free(tensor: *mut DgTensor) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if !tensor.is_null() {
            // SAFETY: the pointer must have been returned by a tensor constructor exactly once.
            unsafe { drop(Box::from_raw(tensor)) };
        }
    }));
}

/// Returns a copied tensor byte snapshot.
#[no_mangle]
pub unsafe extern "C" fn dg_tensor_data(
    tensor: *const DgTensor,
    out_data: *mut *const u8,
    out_length: *mut usize,
) -> DgStatus {
    match ffi_result(|| {
        if tensor.is_null() || out_data.is_null() || out_length.is_null() {
            return Err((
                DgStatus::NullPointer,
                "tensor data argument is null".to_string(),
            ));
        }
        let tensor = unsafe { &*tensor };
        let snapshot = tensor.tensor.buffer().read_bytes().into_boxed_slice();
        let length = snapshot.len();
        LAST_DATA.with(|last| *last.borrow_mut() = Some(snapshot));
        let data = LAST_DATA.with(|last| {
            last.borrow()
                .as_ref()
                .map_or(ptr::null(), |snapshot| snapshot.as_ptr())
        });
        // SAFETY: output pointers were checked non-null and point to writable storage.
        unsafe {
            out_data.write(data);
            out_length.write(length);
        }
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Pushes one tensor into the built graph.
#[no_mangle]
pub unsafe extern "C" fn dg_engine_push(
    engine: *mut DgEngine,
    tensor: *const DgTensor,
) -> DgStatus {
    match ffi_result(|| {
        if engine.is_null() || tensor.is_null() {
            return Err((
                DgStatus::NullPointer,
                "engine or tensor pointer is null".to_string(),
            ));
        }
        let engine = unsafe { &mut *engine };
        let tensor = unsafe { &*tensor };
        engine
            .inner
            .push(tensor.tensor.clone())
            .map_err(|message| {
                if message.contains("input node") {
                    (DgStatus::Unsupported, message)
                } else if message.contains("built") {
                    (DgStatus::NotBuilt, message)
                } else {
                    (DgStatus::RuntimeError, message)
                }
            })?;
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Polls one output tensor. `Again` means the queue is empty.
#[no_mangle]
pub unsafe extern "C" fn dg_engine_poll(
    engine: *mut DgEngine,
    out: *mut *mut DgTensor,
) -> DgStatus {
    match ffi_result(|| {
        if engine.is_null() || out.is_null() {
            return Err((
                DgStatus::NullPointer,
                "engine or output pointer is null".to_string(),
            ));
        }
        let engine = unsafe { &mut *engine };
        let tensor = engine
            .inner
            .outputs
            .pop_front()
            .ok_or((DgStatus::Again, "no output is available".to_string()))?;
        // SAFETY: `out` was checked non-null and points to writable caller storage.
        unsafe { out.write(Box::into_raw(Box::new(DgTensor { tensor }))) };
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Imports an external buffer handle without dereferencing the external address.
#[no_mangle]
pub unsafe extern "C" fn dg_buffer_import_external(
    fd: c_int,
    raw: u64,
    domain: DgMemoryDomain,
    device: DgDeviceKind,
    size_bytes: usize,
    out: *mut *mut DgBuffer,
) -> DgStatus {
    match ffi_result(|| {
        if out.is_null() {
            return Err((
                DgStatus::NullPointer,
                "buffer output pointer is null".to_string(),
            ));
        }
        let external = if fd >= 0 {
            ExternalHandle::from_fd(fd)
        } else {
            ExternalHandle::from_raw(raw)
        };
        let buffer = Buffer::from_external(
            device_from_c(device),
            domain_from_c(domain),
            BufferDesc::new(size_bytes, 1),
            external,
            vec![0; size_bytes],
            ExternalDropGuard::new(|| {}),
        )
        .map_err(|error| (DgStatus::RuntimeError, error.to_string()))?;
        // SAFETY: `out` was checked non-null and points to writable caller storage.
        unsafe { out.write(Box::into_raw(Box::new(DgBuffer { buffer }))) };
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Returns the logical size of an imported buffer.
#[no_mangle]
pub unsafe extern "C" fn dg_buffer_size(buffer: *const DgBuffer, out_size: *mut usize) -> DgStatus {
    match ffi_result(|| {
        if buffer.is_null() || out_size.is_null() {
            return Err((
                DgStatus::NullPointer,
                "buffer or size pointer is null".to_string(),
            ));
        }
        let buffer = unsafe { &*buffer };
        // SAFETY: `out_size` was checked non-null and points to writable caller storage.
        unsafe { out_size.write(buffer.buffer.len()) };
        Ok(())
    }) {
        Ok(()) => DgStatus::Ok,
        Err(status) => status,
    }
}

/// Frees an external buffer handle.
#[no_mangle]
pub unsafe extern "C" fn dg_buffer_free(buffer: *mut DgBuffer) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if !buffer.is_null() {
            // SAFETY: the pointer must have been returned by `dg_buffer_import_external`.
            unsafe { drop(Box::from_raw(buffer)) };
        }
    }));
}

/// Returns the package version as a Rust string for compatibility with M0.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn graph_spec() -> CString {
        CString::new(
            r#"apiVersion: dg/v1
kind: Graph
nodes:
  - name: input
    kind: input
    params: {}
  - name: infer
    kind: mock_inference
    params:
      shape: [1, 4]
      echo_inputs: true
  - name: sink
    kind: sink
    params: {}
connections:
  - input.out -> infer.in
  - infer.out -> sink.in
"#,
        )
        .expect("valid graph spec")
    }

    fn unique_temp_path() -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        std::env::temp_dir().join(format!("dg-capi-invalid-{nanos}.yaml"))
    }

    fn updated_graph_spec() -> CString {
        CString::new(
            r#"apiVersion: dg/v1
kind: Graph
nodes:
  - name: input
    kind: input
    params: {}
  - name: infer
    kind: mock_inference
    params:
      shape: [1, 4]
      echo_inputs: false
      fill_value: 7
  - name: sink
    kind: sink
    params: {}
  - name: extra_source
    kind: source
    params:
      count: 0
      shape: [1, 4]
  - name: extra_sink
    kind: sink
    params: {}
connections:
  - input.out -> infer.in
  - infer.out -> sink.in
  - extra_source.out -> extra_sink.in
"#,
        )
        .expect("valid updated graph spec")
    }

    #[test]
    fn c_abi_push_poll_round_trip() {
        let mut engine = ptr::null_mut();
        assert_eq!(unsafe { dg_engine_create(&mut engine) }, DgStatus::Ok);
        let spec = graph_spec();
        assert_eq!(
            unsafe { dg_engine_load_string(engine, DgGraphFormat::Yaml, spec.as_ptr()) },
            DgStatus::Ok
        );
        assert_eq!(unsafe { dg_engine_build(engine) }, DgStatus::Ok);

        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let input_bytes: Vec<u8> = input.iter().flat_map(|value| value.to_ne_bytes()).collect();
        let shape = [1_usize, 4];
        let mut tensor = ptr::null_mut();
        assert_eq!(
            unsafe {
                dg_tensor_create(
                    input_bytes.as_ptr(),
                    input_bytes.len(),
                    shape.as_ptr(),
                    shape.len(),
                    DgDataType::F32,
                    DgDataFormat::Nc,
                    DgDeviceKind::Cpu,
                    &mut tensor,
                )
            },
            DgStatus::Ok
        );
        assert_eq!(unsafe { dg_engine_push(engine, tensor) }, DgStatus::Ok);
        let run_status = unsafe { dg_engine_run(engine) };
        let error = dg_last_error();
        let error = if error.is_null() {
            "<missing last error>".to_string()
        } else {
            unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned()
        };
        assert_eq!(run_status, DgStatus::Ok, "{}", error);
        let mut output = ptr::null_mut();
        assert_eq!(unsafe { dg_engine_poll(engine, &mut output) }, DgStatus::Ok);
        let mut output_data = ptr::null();
        let mut output_len = 0;
        assert_eq!(
            unsafe { dg_tensor_data(output, &mut output_data, &mut output_len) },
            DgStatus::Ok
        );
        assert_eq!(
            unsafe { std::slice::from_raw_parts(output_data, output_len) },
            input_bytes.as_slice()
        );
        unsafe {
            dg_tensor_free(output);
            dg_tensor_free(tensor);
            dg_engine_free(engine);
        }
    }

    #[test]
    fn invalid_pointer_returns_status_and_error() {
        let status = unsafe { dg_engine_build(ptr::null_mut()) };
        assert_eq!(status, DgStatus::NullPointer);
        assert!(!dg_last_error().is_null());
    }

    #[test]
    fn load_file_rejects_invalid_graph_during_load() {
        let path = unique_temp_path();
        fs::write(
            &path,
            r#"apiVersion: dg/v1
kind: Graph
nodes:
  - name: duplicate
    kind: source
    params: {count: 0}
  - name: duplicate
    kind: sink
    params: {}
connections: []
"#,
        )
        .expect("write invalid graph");
        let path_string =
            CString::new(path.to_str().expect("temp path is utf8")).expect("temp path has no nul");
        let mut engine = ptr::null_mut();
        assert_eq!(unsafe { dg_engine_create(&mut engine) }, DgStatus::Ok);
        assert_eq!(
            unsafe { dg_engine_load_file(engine, path_string.as_ptr()) },
            DgStatus::ParseError
        );
        assert!(!dg_last_error().is_null());
        unsafe { dg_engine_free(engine) };
        fs::remove_file(path).expect("remove invalid graph");
    }

    #[test]
    fn c_abi_diff_and_reload_invalidate_built_graph() {
        let mut engine = ptr::null_mut();
        assert_eq!(unsafe { dg_engine_create(&mut engine) }, DgStatus::Ok);
        let initial = graph_spec();
        let updated = updated_graph_spec();
        assert_eq!(
            unsafe { dg_engine_load_string(engine, DgGraphFormat::Yaml, initial.as_ptr()) },
            DgStatus::Ok
        );
        assert_eq!(unsafe { dg_engine_build(engine) }, DgStatus::Ok);

        let mut added_nodes = 0;
        let mut removed_nodes = 0;
        let mut updated_nodes = 0;
        let mut added_connections = 0;
        let mut removed_connections = 0;
        assert_eq!(
            unsafe {
                dg_engine_diff_string(
                    engine,
                    DgGraphFormat::Yaml,
                    updated.as_ptr(),
                    &mut added_nodes,
                    &mut removed_nodes,
                    &mut updated_nodes,
                    &mut added_connections,
                    &mut removed_connections,
                )
            },
            DgStatus::Ok
        );
        assert_eq!(added_nodes, 2);
        assert_eq!(removed_nodes, 0);
        assert_eq!(updated_nodes, 1);
        assert_eq!(added_connections, 1);
        assert_eq!(removed_connections, 0);

        let invalid = CString::new("not a graph").expect("valid invalid spec bytes");
        assert_eq!(
            unsafe { dg_engine_reload_string(engine, DgGraphFormat::Yaml, invalid.as_ptr()) },
            DgStatus::ParseError
        );
        assert!(!dg_last_error().is_null());
        assert_eq!(
            unsafe { dg_engine_reload_string(engine, DgGraphFormat::Yaml, updated.as_ptr()) },
            DgStatus::Ok
        );
        assert_eq!(unsafe { dg_engine_run(engine) }, DgStatus::NotBuilt);
        unsafe { dg_engine_free(engine) };
    }

    #[test]
    fn external_buffer_import_preserves_handle_metadata() {
        let mut buffer = ptr::null_mut();
        assert_eq!(
            unsafe {
                dg_buffer_import_external(
                    -1,
                    0x1234,
                    DgMemoryDomain::CudaDevice,
                    DgDeviceKind::CudaGpu,
                    16,
                    &mut buffer,
                )
            },
            DgStatus::Ok
        );
        let mut size = 0;
        assert_eq!(unsafe { dg_buffer_size(buffer, &mut size) }, DgStatus::Ok);
        assert_eq!(size, 16);
        unsafe { dg_buffer_free(buffer) };
    }
}
