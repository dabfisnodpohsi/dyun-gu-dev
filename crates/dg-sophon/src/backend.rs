//! Feature-gated Sophon (Sophgo BM series) backend built on BMRuntime.
//!
//! The execution flow mirrors the vendor recipe from `docs/design.md` (Sophon
//! backend adapter): `bmrt_create → bmrt_load_bmodel_data →
//! bmrt_launch_tensor_ex`, with `bmlib`
//! managing device memory (`bm_malloc_device_byte` / `bm_free_device` /
//! `bm_memcpy_*`) and `bm_thread_sync` for completion. Every C resource is owned
//! by an RAII wrapper so it is released exactly once, and `unsafe` is confined to
//! the FFI calls with a stated safety invariant.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

use dg_core::{CpuDevice, Shape, Tensor};
use dg_runtime::{
    backend_capabilities, supports_precision, BackendConfig, BackendDescriptor, BackendKind,
    BackendOptions, Error, InferBackend, ModelSource, Result, RuntimeCapabilities, RuntimeOption,
    SophonOptions, TensorInfo,
};
use serde::Deserialize;
use tracing::{debug, trace};

#[cfg(not(feature = "backend"))]
use crate::mock_sys as sys;
#[cfg(feature = "backend")]
use dg_sophon_sys as sys;

use crate::convert::{self, SophonDataType};
use crate::validate::{validate_deploy_mode, validate_options};

/// Returns `true` when the real Sophon runtime is compiled in.
#[cfg_attr(test, allow(dead_code))]
pub const fn backend_enabled() -> bool {
    true
}

/// Deployment mode this crate was compiled for. SoC and Host (PCIe) builds link
/// against different vendor libraries, so the runtime request must match.
const fn compiled_deploy_mode() -> dg_core::DeployMode {
    if cfg!(feature = "soc") {
        dg_core::DeployMode::SoC
    } else {
        dg_core::DeployMode::Host
    }
}

struct ModelBuffer {
    data: Vec<u8>,
}

impl ModelBuffer {
    fn new(data: Vec<u8>) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::InvalidOption("Sophon bmodel is empty".to_string()));
        }
        Ok(Self { data })
    }

    fn as_ptr(&self) -> *const c_void {
        self.data.as_ptr().cast()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

/// RAII wrapper around a `bmlib` device memory allocation.
struct DeviceMem {
    handle: sys::bm_handle_t,
    mem: sys::bm_device_mem_t,
    owned: bool,
}

impl DeviceMem {
    fn alloc(handle: sys::bm_handle_t, size: usize) -> Result<Self> {
        let bytes = u32::try_from(size)
            .map_err(|_| Error::InvalidOption("Sophon allocation exceeds u32".to_string()))?;
        // SAFETY: `handle` is a live device handle and `mem` is a valid out
        // pointer; on success bmlib fills it with an owned allocation.
        let mut mem: sys::bm_device_mem_t = unsafe { std::mem::zeroed() };
        let status = unsafe { sys::bm_malloc_device_byte(handle, &mut mem, bytes) };
        check_status(status, "bm_malloc_device_byte")?;
        Ok(Self {
            handle,
            mem,
            owned: true,
        })
    }

    fn from_external(handle: sys::bm_handle_t, address: u64, size: usize) -> Result<Self> {
        let size = u64::try_from(size)
            .map_err(|_| Error::InvalidOption("Sophon buffer exceeds u64".to_string()))?;
        // SAFETY: the address and size come from the caller-owned external
        // buffer, which remains alive through the synchronous launch.
        let mem = unsafe { sys::bm_mem_from_device(address, size) };
        if mem.ptr.is_null() || mem.size < usize::try_from(size).unwrap_or(usize::MAX) {
            return Err(Error::Backend(
                "bm_mem_from_device returned an invalid device buffer".to_string(),
            ));
        }
        Ok(Self {
            handle,
            mem,
            owned: false,
        })
    }

    fn upload(&self, src: &[u8]) -> Result<()> {
        // SAFETY: `src` outlives the synchronous copy and the allocation was
        // sized to hold at least `src.len()` bytes.
        let status =
            unsafe { sys::bm_memcpy_s2d(self.handle, self.mem, src.as_ptr() as *mut c_void) };
        check_status(status, "bm_memcpy_s2d")
    }

    fn download(&self, dst: &mut [u8]) -> Result<()> {
        // SAFETY: `dst` outlives the synchronous copy and is sized to receive
        // the requested number of bytes from the allocation.
        let status = unsafe { sys::bm_memcpy_d2s(self.handle, dst.as_mut_ptr().cast(), self.mem) };
        check_status(status, "bm_memcpy_d2s")
    }
}

impl Drop for DeviceMem {
    fn drop(&mut self) {
        // SAFETY: `mem` was produced by `bm_malloc_device_byte` on this handle
        // and is freed exactly once here.
        if self.owned {
            unsafe { sys::bm_free_device(self.handle, self.mem) };
        }
    }
}

pub struct SophonBackend {
    handle: Option<sys::bm_handle_t>,
    runtime: Option<*mut c_void>,
    net_name: Option<CString>,
    options: SophonOptions,
    input_infos: Vec<TensorInfo>,
    output_infos: Vec<TensorInfo>,
    last_copy_count: usize,
}

// SAFETY: the raw BMRuntime handle and runtime pointer are only reachable
// through `&mut self`, so the backend is used by one thread at a time and owns
// its resources uniquely. It is never shared, so only `Send` (not `Sync`) is
// asserted, matching the `InferBackend: Send` contract.
unsafe impl Send for SophonBackend {}

impl SophonBackend {
    fn new() -> Self {
        Self {
            handle: None,
            runtime: None,
            net_name: None,
            options: SophonOptions::default(),
            input_infos: Vec::new(),
            output_infos: Vec::new(),
            last_copy_count: 0,
        }
    }

    fn handle(&self) -> Result<sys::bm_handle_t> {
        self.handle
            .ok_or_else(|| Error::BackendUnavailable("Sophon device not initialized".to_string()))
    }

    fn runtime(&self) -> Result<*mut c_void> {
        self.runtime
            .ok_or_else(|| Error::BackendUnavailable("Sophon runtime not initialized".to_string()))
    }

    fn net_name(&self) -> Result<&CString> {
        self.net_name
            .as_ref()
            .ok_or_else(|| Error::BackendUnavailable("Sophon network not loaded".to_string()))
    }

    fn load_model(source: &ModelSource) -> Result<ModelBuffer> {
        match source {
            ModelSource::File(path) => {
                let data = std::fs::read(path).map_err(|err| {
                    Error::BackendUnavailable(format!(
                        "failed to read Sophon bmodel {}: {err}",
                        path.display()
                    ))
                })?;
                ModelBuffer::new(data)
            }
            ModelSource::Bytes(bytes) => ModelBuffer::new(bytes.clone()),
        }
    }

    /// Queries the loaded bmodel for its first network and records the static
    /// input/output tensor metadata (names, dtypes, shapes from stage 0).
    fn discover_network(&mut self) -> Result<()> {
        let p_bmrt = self.runtime()?;
        let count = unsafe { sys::bmrt_get_network_number(p_bmrt) };
        if count <= 0 {
            return Err(Error::BackendUnavailable(
                "Sophon bmodel exposes no networks".to_string(),
            ));
        }

        let mut names: *mut *const c_char = ptr::null_mut();
        // SAFETY: bmrt allocates the name array and writes it through the out
        // pointer; ownership transfers to us and is released below.
        unsafe { sys::bmrt_get_network_names(p_bmrt, &mut names) };
        if names.is_null() {
            return Err(Error::Backend(
                "bmrt_get_network_names returned null".to_string(),
            ));
        }
        // SAFETY: `names` points to at least one valid, NUL-terminated string.
        let first = unsafe { *names };
        let net_name = if first.is_null() {
            // SAFETY: array allocated by bmrt with the C allocator.
            unsafe { free_c(names.cast()) };
            return Err(Error::Backend("Sophon network name is null".to_string()));
        } else {
            // SAFETY: `first` is a valid NUL-terminated C string owned by bmrt.
            let owned = unsafe { CStr::from_ptr(first) }.to_owned();
            // SAFETY: array allocated by bmrt with the C allocator; the copied
            // name above no longer references it.
            unsafe { free_c(names.cast()) };
            owned
        };

        // SAFETY: `net_name` is a valid C string; bmrt returns a pointer into
        // runtime-owned storage that lives as long as the loaded model.
        let info_ptr = unsafe { sys::bmrt_get_network_info(p_bmrt, net_name.as_ptr()) };
        if info_ptr.is_null() {
            return Err(Error::Backend(
                "bmrt_get_network_info returned null".to_string(),
            ));
        }
        // SAFETY: non-null pointer to a valid `bm_net_info_t` owned by bmrt.
        let info = unsafe { &*info_ptr };
        if info.stage_num <= 0 || info.stages.is_null() {
            return Err(Error::Backend(
                "Sophon network reports no stages".to_string(),
            ));
        }
        // SAFETY: `stages` points to at least one `bm_stage_info_t`.
        let stage = unsafe { &*info.stages };

        let input_num = usize::try_from(info.input_num)
            .map_err(|_| Error::Backend("Sophon input count is negative".to_string()))?;
        let output_num = usize::try_from(info.output_num)
            .map_err(|_| Error::Backend("Sophon output count is negative".to_string()))?;

        let inputs = collect_infos(
            input_num,
            info.input_dtypes,
            info.input_names,
            stage.input_shapes,
        )?;
        let outputs = collect_infos(
            output_num,
            info.output_dtypes,
            info.output_names,
            stage.output_shapes,
        )?;

        for info in inputs.iter().chain(outputs.iter()) {
            if !supports_precision(BackendKind::Sophon, info.dtype) {
                return Err(Error::UnsupportedPrecision(info.dtype));
            }
        }

        debug!(
            network = %net_name.to_string_lossy(),
            inputs = inputs.len(),
            outputs = outputs.len(),
            "discovered Sophon network"
        );
        self.net_name = Some(net_name);
        self.input_infos = inputs;
        self.output_infos = outputs;
        Ok(())
    }

    fn build_bm_tensor(
        dtype: SophonDataType,
        shape: &Shape,
        mem: sys::bm_device_mem_t,
    ) -> Result<sys::bm_tensor_t> {
        let (num_dims, dims) = convert::bm_shape_dims(shape)?;
        // SAFETY: `bm_tensor_t` is a C POD struct; a zeroed value is a valid
        // initial state before we populate the meaningful fields.
        let mut tensor: sys::bm_tensor_t = unsafe { std::mem::zeroed() };
        tensor.dtype = dtype.bm_code();
        tensor.shape.num_dims = num_dims;
        tensor.shape.dims.copy_from_slice(&dims);
        tensor.device_mem = mem;
        tensor.st_mode = 0;
        Ok(tensor)
    }
}

impl InferBackend for SophonBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Sophon
    }

    fn probe_capabilities(&self) -> Result<RuntimeCapabilities> {
        backend_capabilities(BackendKind::Sophon)
            .map(RuntimeCapabilities::from_static)
            .ok_or_else(|| Error::BackendUnavailable("Sophon capabilities unavailable".to_string()))
    }

    fn init(&mut self, option: &RuntimeOption) -> Result<()> {
        trace!("initializing Sophon backend");
        let BackendOptions::Sophon(sophon) = &option.backend_options else {
            return Err(Error::InvalidOption(
                "Sophon backend requires Sophon backend options".to_string(),
            ));
        };
        validate_options(option, sophon)?;
        validate_deploy_mode(sophon.deploy_mode, compiled_deploy_mode())?;
        self.options = sophon.clone();

        let model = Self::load_model(&option.model_source)?;

        let device_id = i32::try_from(self.options.device_id.unwrap_or(0))
            .map_err(|_| Error::InvalidOption("Sophon device id overflows i32".to_string()))?;
        let mut handle: sys::bm_handle_t = ptr::null_mut();
        // SAFETY: `handle` is a valid out pointer; on success bmlib returns an
        // owned device handle freed in `Drop`.
        let status = unsafe { sys::bm_dev_request(&mut handle, device_id) };
        check_status(status, "bm_dev_request")?;
        self.handle = Some(handle);

        // SAFETY: `handle` is a live device handle just acquired above.
        let runtime = unsafe { sys::bmrt_create(handle) };
        if runtime.is_null() {
            return Err(Error::BackendUnavailable(
                "bmrt_create returned null".to_string(),
            ));
        }
        self.runtime = Some(runtime);

        // SAFETY: `runtime` is a live BMRuntime instance and the model buffer
        // outlives the call.
        let loaded = unsafe { sys::bmrt_load_bmodel_data(runtime, model.as_ptr(), model.len()) };
        if !loaded {
            return Err(Error::BackendUnavailable(
                "bmrt_load_bmodel_data failed".to_string(),
            ));
        }

        self.discover_network()
    }

    fn reshape(&mut self, input_shapes: &[Shape]) -> Result<()> {
        if input_shapes.len() != self.input_infos.len() {
            return Err(Error::InvalidOption(format!(
                "Sophon reshape expected {} input shapes, got {}",
                self.input_infos.len(),
                input_shapes.len()
            )));
        }
        for (info, shape) in self.input_infos.iter_mut().zip(input_shapes) {
            info.shape = shape.clone();
        }
        Ok(())
    }

    fn input_count(&self) -> usize {
        self.input_infos.len()
    }

    fn output_count(&self) -> usize {
        self.output_infos.len()
    }

    fn input_info(&self, index: usize) -> Result<&TensorInfo> {
        self.input_infos
            .get(index)
            .ok_or_else(|| Error::InvalidOption(format!("input index out of range: {index}")))
    }

    fn output_info(&self, index: usize) -> Result<&TensorInfo> {
        self.output_infos
            .get(index)
            .ok_or_else(|| Error::InvalidOption(format!("output index out of range: {index}")))
    }

    fn input_infos(&self) -> &[TensorInfo] {
        &self.input_infos
    }

    fn output_infos(&self) -> &[TensorInfo] {
        &self.output_infos
    }

    fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        self.last_copy_count = inputs.len();
        let handle = self.handle()?;
        let p_bmrt = self.runtime()?;
        if inputs.len() != self.input_infos.len() {
            return Err(Error::InvalidOption(format!(
                "Sophon run expected {} inputs, got {}",
                self.input_infos.len(),
                inputs.len()
            )));
        }

        let mut input_mems: Vec<DeviceMem> = Vec::with_capacity(inputs.len());
        let mut input_tensors: Vec<sys::bm_tensor_t> = Vec::with_capacity(inputs.len());
        for (index, tensor) in inputs.iter().enumerate() {
            let info = &self.input_infos[index];
            let dtype = SophonDataType::from_data_type(info.dtype)?;
            let shape = tensor.desc().shape().clone();
            let expected = convert::byte_size(dtype, &shape)?;
            let external = tensor.buffer().external();
            let (mem, direct) = if tensor.buffer().domain() == dg_core::MemoryDomain::SophonDevice
                && external.raw != 0
            {
                (
                    DeviceMem::from_external(handle, external.raw, expected)?,
                    true,
                )
            } else {
                let host = tensor.buffer().try_read_bytes()?;
                if host.len() != expected {
                    return Err(Error::InvalidOption(format!(
                        "Sophon input {index} byte size mismatch: expected {expected}, got {}",
                        host.len()
                    )));
                }
                let mem = DeviceMem::alloc(handle, expected)?;
                mem.upload(&host)?;
                (mem, false)
            };
            if direct {
                self.last_copy_count = self.last_copy_count.saturating_sub(1);
            }
            input_tensors.push(Self::build_bm_tensor(dtype, &shape, mem.mem)?);
            input_mems.push(mem);
        }

        let mut output_mems: Vec<DeviceMem> = Vec::with_capacity(self.output_infos.len());
        let mut output_tensors: Vec<sys::bm_tensor_t> = Vec::with_capacity(self.output_infos.len());
        for info in &self.output_infos {
            let dtype = SophonDataType::from_data_type(info.dtype)?;
            let size = convert::byte_size(dtype, &info.shape)?;
            let mem = DeviceMem::alloc(handle, size)?;
            output_tensors.push(Self::build_bm_tensor(dtype, &info.shape, mem.mem)?);
            output_mems.push(mem);
        }

        let input_num = i32::try_from(input_tensors.len())
            .map_err(|_| Error::InvalidOption("too many Sophon inputs".to_string()))?;
        let output_num = i32::try_from(output_tensors.len())
            .map_err(|_| Error::InvalidOption("too many Sophon outputs".to_string()))?;

        // SAFETY: the tensor slices live for the duration of the call, device
        // memory is user-owned (`user_mem = true`), and the network name is a
        // valid C string. `user_stmode = false` keeps the default store mode.
        let launched = unsafe {
            sys::bmrt_launch_tensor_ex(
                p_bmrt,
                self.net_name()?.as_ptr(),
                input_tensors.as_ptr(),
                input_num,
                output_tensors.as_mut_ptr(),
                output_num,
                true,
                false,
            )
        };
        if !launched {
            return Err(Error::Backend("bmrt_launch_tensor_ex failed".to_string()));
        }

        // SAFETY: `handle` is the live device handle used for the launch above.
        let status = unsafe { sys::bm_thread_sync(handle) };
        check_status(status, "bm_thread_sync")?;
        debug!(
            backend = "sophon",
            copy_count = self.last_copy_count,
            path = "staging",
            "Sophon input binding completed"
        );

        let device = CpuDevice::new();
        let mut results = Vec::with_capacity(output_tensors.len());
        for (index, out) in output_tensors.iter().enumerate() {
            let code = i32::try_from(out.dtype)
                .map_err(|_| Error::Backend("Sophon output dtype overflow".to_string()))?;
            let dtype = SophonDataType::from_code(code)?;
            let shape = convert::shape_from_bm(out.shape.num_dims, &out.shape.dims)?;
            let size = convert::byte_size(dtype, &shape)?;
            let mut host = vec![0u8; size];
            output_mems[index].download(&mut host)?;

            let mut info = TensorInfo::new(shape, dtype.to_data_type());
            if let Some(name) = &self.output_infos[index].name {
                info = info.with_name(name.clone());
            }
            if let Some(layout) = self.output_infos[index].layout {
                info = info.with_layout(layout);
            }
            let tensor = info.allocate(&device)?;
            tensor.buffer().write_from_slice(&host)?;
            results.push(tensor);
        }

        Ok(results)
    }

    fn last_copy_count(&self) -> usize {
        self.last_copy_count
    }
}

impl Drop for SophonBackend {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            // SAFETY: `runtime` came from `bmrt_create` and is destroyed once.
            unsafe { sys::bmrt_destroy(runtime) };
        }
        if let Some(handle) = self.handle.take() {
            // SAFETY: `handle` came from `bm_dev_request` and is freed once,
            // after the runtime that used it has been destroyed above.
            unsafe { sys::bm_dev_free(handle) };
        }
    }
}

fn collect_infos(
    count: usize,
    dtypes: *const sys::bm_data_type_t,
    names: *const *const c_char,
    shapes: *const sys::bm_shape_t,
) -> Result<Vec<TensorInfo>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if dtypes.is_null() || shapes.is_null() {
        return Err(Error::Backend(
            "Sophon network metadata is incomplete".to_string(),
        ));
    }
    let mut infos = Vec::with_capacity(count);
    for index in 0..count {
        // SAFETY: bmrt guarantees the dtype/shape arrays hold `count` entries.
        let code = i32::try_from(unsafe { *dtypes.add(index) })
            .map_err(|_| Error::Backend("Sophon dtype code overflow".to_string()))?;
        let dtype = SophonDataType::from_code(code)?;
        // SAFETY: shape array holds `count` valid entries.
        let bm_shape = unsafe { &*shapes.add(index) };
        let shape = convert::shape_from_bm(bm_shape.num_dims, &bm_shape.dims)?;
        let mut info = TensorInfo::new(shape, dtype.to_data_type());
        if !names.is_null() {
            // SAFETY: name array (when present) holds `count` C-string pointers.
            let name_ptr = unsafe { *names.add(index) };
            if !name_ptr.is_null() {
                // SAFETY: `name_ptr` is a valid NUL-terminated C string.
                if let Ok(name) = unsafe { CStr::from_ptr(name_ptr) }.to_str() {
                    if !name.is_empty() {
                        info = info.with_name(name.to_string());
                    }
                }
            }
        }
        infos.push(info);
    }
    Ok(infos)
}

fn check_status(status: sys::bm_status_t, context: &str) -> Result<()> {
    if status == sys::bm_status_t::BM_SUCCESS {
        Ok(())
    } else {
        Err(Error::Backend(format!(
            "{context} failed with Sophon status {status:?}"
        )))
    }
}

/// Releases a buffer allocated by the C runtime (e.g. `bmrt_get_network_names`).
///
/// # Safety
/// `ptr` must be a pointer returned by the vendor library's allocator, or null.
unsafe fn free_c(ptr: *mut c_void) {
    #[cfg(feature = "backend")]
    if !ptr.is_null() {
        libc::free(ptr);
    }
    #[cfg(not(feature = "backend"))]
    crate::mock_sys::free_c(ptr);
}

fn create_sophon_backend() -> Box<dyn InferBackend> {
    Box::new(SophonBackend::new())
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct SophonConfig {
    device_id: Option<u32>,
}

fn configure_sophon(config: BackendConfig) -> Result<RuntimeOption> {
    let params: SophonConfig = config.parse_options("sophon")?;
    let model_source = config.require_model_file("Sophon")?;
    let deploy_mode = config.deploy_mode().unwrap_or(dg_core::DeployMode::Host);
    let core_mask = config.core_mask();
    Ok(config.into_runtime_option(
        BackendKind::Sophon,
        model_source,
        BackendOptions::Sophon(SophonOptions {
            deploy_mode,
            device_id: params.device_id,
            core_mask,
        }),
    ))
}

inventory::submit! {
    BackendDescriptor {
        kind: BackendKind::Sophon,
        name: "sophon",
        create: create_sophon_backend,
        configure: configure_sophon,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock_sys;

    fn option() -> RuntimeOption {
        RuntimeOption::new(
            BackendKind::Sophon,
            ModelSource::Bytes(mock_sys::encode_mock_model()),
            BackendOptions::Sophon(SophonOptions::default()),
        )
    }

    #[test]
    fn mock_adapter_initializes_and_inspects_network() {
        let mut backend = SophonBackend::new();
        backend.init(&option()).expect("mock Sophon init");
        assert_eq!(backend.input_count(), 1);
        assert_eq!(backend.output_count(), 1);
        assert_eq!(backend.input_info(0).expect("input").shape.dims(), &[1, 4]);
    }

    #[test]
    fn mock_adapter_runs_round_trip_and_rejects_invalid_inputs() {
        let mut backend = SophonBackend::new();
        backend.init(&option()).expect("mock Sophon init");
        let info = backend.input_info(0).expect("input").clone();
        let device = dg_core::CpuDevice::new();
        let input = info.allocate(&device).expect("input allocation");
        input
            .buffer()
            .write_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            .expect("input bytes");
        let outputs = backend.run(std::slice::from_ref(&input)).expect("run");
        assert!(backend.last_copy_count() > 0);
        assert_eq!(
            outputs[0].buffer().read_bytes(),
            input.buffer().read_bytes()
        );
        assert!(matches!(backend.run(&[]), Err(Error::InvalidOption(_))));
    }

    #[test]
    fn device_external_input_binds_without_copy() {
        let mut backend = SophonBackend::new();
        backend.init(&option()).expect("mock Sophon init");
        let mut bytes = (0..16).collect::<Vec<u8>>();
        let pointer = bytes.as_mut_ptr() as usize;
        let buffer = dg_core::Buffer::from_external(
            dg_core::DeviceKind::SophonTpu,
            dg_core::MemoryDomain::SophonDevice,
            dg_core::BufferDesc::new(16, 1),
            dg_core::ExternalHandle::from_raw(pointer as u64),
            dg_core::ExternalDropGuard::new(move || {
                // SAFETY: ownership of this allocation is transferred to the
                // external guard exactly once.
                drop(unsafe { Vec::from_raw_parts(pointer as *mut u8, 16, 16) });
            }),
        )
        .expect("external Sophon memory");
        std::mem::forget(bytes);
        let input = dg_core::Tensor::from_buffer(
            dg_core::TensorDesc::new(
                dg_core::Shape::new([1, 4]),
                dg_core::DataType::F32,
                dg_core::DataFormat::Auto,
                dg_core::DeviceKind::SophonTpu,
            ),
            buffer,
        )
        .expect("tensor");
        let outputs = backend.run(&[input]).expect("run");
        assert_eq!(backend.last_copy_count(), 0);
        assert_eq!(
            outputs[0].buffer().read_bytes(),
            (0..16).collect::<Vec<_>>()
        );
    }
}
