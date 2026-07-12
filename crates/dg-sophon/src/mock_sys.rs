//! SDK-free Sophon ABI shim used by the backend adapter tests.

#![allow(
    dead_code,
    non_camel_case_types,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

use std::ffi::{c_char, c_void, CString};
use std::ptr;

use crate::convert::BM_MAX_DIMS;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum bm_status_t {
    BM_SUCCESS = 0,
    BM_ERR_FAILURE = -1,
}

pub type bm_data_type_t = u32;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct bm_shape_t {
    pub num_dims: i32,
    pub dims: [i32; BM_MAX_DIMS],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct bm_device_mem_t {
    pub ptr: *mut u8,
    pub size: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct bm_tensor_t {
    pub dtype: u32,
    pub shape: bm_shape_t,
    pub device_mem: bm_device_mem_t,
    pub st_mode: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct bm_stage_info_t {
    pub input_shapes: *const bm_shape_t,
    pub output_shapes: *const bm_shape_t,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct bm_net_info_t {
    pub stage_num: i32,
    pub stages: *const bm_stage_info_t,
    pub input_num: i32,
    pub output_num: i32,
    pub input_dtypes: *const bm_data_type_t,
    pub output_dtypes: *const bm_data_type_t,
    pub input_names: *const *const c_char,
    pub output_names: *const *const c_char,
}

pub type bm_handle_t = *mut MockDevice;

pub struct MockDevice;

struct MockRuntime {
    name: CString,
    input_dtypes: [bm_data_type_t; 1],
    output_dtypes: [bm_data_type_t; 1],
    input_name: [*const c_char; 1],
    output_name: [*const c_char; 1],
    input_shapes: [bm_shape_t; 1],
    output_shapes: [bm_shape_t; 1],
    stage: bm_stage_info_t,
    info: bm_net_info_t,
}

impl MockRuntime {
    fn new() -> Box<Self> {
        let name = CString::new("mock_network").expect("static name");
        let input_name = [name.as_ptr()];
        let output_name = [name.as_ptr()];
        let input_shapes = [shape()];
        let output_shapes = [shape()];
        let stage = bm_stage_info_t {
            input_shapes: input_shapes.as_ptr(),
            output_shapes: output_shapes.as_ptr(),
        };
        let mut runtime = Box::new(Self {
            name,
            input_dtypes: [0],
            output_dtypes: [0],
            input_name,
            output_name,
            input_shapes,
            output_shapes,
            stage,
            info: bm_net_info_t {
                stage_num: 1,
                stages: ptr::null(),
                input_num: 1,
                output_num: 1,
                input_dtypes: ptr::null(),
                output_dtypes: ptr::null(),
                input_names: ptr::null(),
                output_names: ptr::null(),
            },
        });
        runtime.stage.input_shapes = runtime.input_shapes.as_ptr();
        runtime.stage.output_shapes = runtime.output_shapes.as_ptr();
        runtime.info.stages = &runtime.stage;
        runtime.info.input_dtypes = runtime.input_dtypes.as_ptr();
        runtime.info.output_dtypes = runtime.output_dtypes.as_ptr();
        runtime.info.input_names = runtime.input_name.as_ptr();
        runtime.info.output_names = runtime.output_name.as_ptr();
        runtime
    }
}

fn shape() -> bm_shape_t {
    bm_shape_t {
        num_dims: 2,
        dims: [1, 4, 0, 0, 0, 0, 0, 0],
    }
}

pub fn encode_mock_model() -> Vec<u8> {
    b"DSOPHONMOCK1".to_vec()
}

pub unsafe fn bm_dev_request(handle: *mut bm_handle_t, _device_id: i32) -> bm_status_t {
    if handle.is_null() {
        return bm_status_t::BM_ERR_FAILURE;
    }
    *handle = Box::into_raw(Box::new(MockDevice));
    bm_status_t::BM_SUCCESS
}

pub unsafe fn bm_dev_free(handle: bm_handle_t) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

pub unsafe fn bmrt_create(_handle: bm_handle_t) -> *mut c_void {
    Box::into_raw(MockRuntime::new()).cast()
}

pub unsafe fn bmrt_destroy(runtime: *mut c_void) {
    if !runtime.is_null() {
        drop(Box::from_raw(runtime.cast::<MockRuntime>()));
    }
}

pub unsafe fn bmrt_load_bmodel_data(
    _runtime: *mut c_void,
    model: *const c_void,
    size: usize,
) -> bool {
    !model.is_null() && size > 0
}

pub unsafe fn bmrt_get_network_number(_runtime: *mut c_void) -> i32 {
    1
}

pub unsafe fn bmrt_get_network_names(runtime: *mut c_void, names: *mut *mut *const c_char) -> bool {
    if runtime.is_null() || names.is_null() {
        return false;
    }
    let runtime = &*runtime.cast::<MockRuntime>();
    let list = Box::new([runtime.name.as_ptr()]);
    *names = Box::into_raw(list).cast();
    true
}

pub unsafe fn bmrt_get_network_info(
    runtime: *mut c_void,
    _name: *const c_char,
) -> *const bm_net_info_t {
    if runtime.is_null() {
        ptr::null()
    } else {
        &(*runtime.cast::<MockRuntime>()).info
    }
}

pub unsafe fn bm_malloc_device_byte(
    _handle: bm_handle_t,
    mem: *mut bm_device_mem_t,
    size: u32,
) -> bm_status_t {
    if mem.is_null() {
        return bm_status_t::BM_ERR_FAILURE;
    }
    let mut bytes = vec![0u8; size as usize];
    let ptr = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    *mem = bm_device_mem_t {
        ptr,
        size: size as usize,
    };
    bm_status_t::BM_SUCCESS
}

pub unsafe fn bm_free_device(_handle: bm_handle_t, mem: bm_device_mem_t) {
    if !mem.ptr.is_null() {
        drop(Vec::from_raw_parts(mem.ptr, mem.size, mem.size));
    }
}

pub unsafe fn bm_memcpy_s2d(
    _handle: bm_handle_t,
    mem: bm_device_mem_t,
    src: *mut c_void,
) -> bm_status_t {
    if mem.ptr.is_null() || src.is_null() {
        return bm_status_t::BM_ERR_FAILURE;
    }
    ptr::copy_nonoverlapping(src.cast::<u8>(), mem.ptr, mem.size);
    bm_status_t::BM_SUCCESS
}

pub unsafe fn bm_memcpy_d2s(
    _handle: bm_handle_t,
    dst: *mut c_void,
    mem: bm_device_mem_t,
) -> bm_status_t {
    if mem.ptr.is_null() || dst.is_null() {
        return bm_status_t::BM_ERR_FAILURE;
    }
    ptr::copy_nonoverlapping(mem.ptr, dst.cast::<u8>(), mem.size);
    bm_status_t::BM_SUCCESS
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn bmrt_launch_tensor_ex(
    _runtime: *mut c_void,
    _name: *const c_char,
    inputs: *const bm_tensor_t,
    input_count: i32,
    outputs: *mut bm_tensor_t,
    output_count: i32,
    _user_mem: bool,
    _user_stmode: bool,
) -> bool {
    if inputs.is_null() || outputs.is_null() || input_count < 1 || output_count < 1 {
        return false;
    }
    let input = &*inputs;
    let output = &mut *outputs;
    if input.device_mem.size != output.device_mem.size {
        return false;
    }
    ptr::copy_nonoverlapping(
        input.device_mem.ptr,
        output.device_mem.ptr,
        input.device_mem.size,
    );
    true
}

pub unsafe fn bm_thread_sync(_handle: bm_handle_t) -> bm_status_t {
    bm_status_t::BM_SUCCESS
}

/// The mock returns static runtime-owned names, so no allocator operation is
/// needed when the safe adapter releases the name list.
pub unsafe fn free_c(_ptr: *mut c_void) {}
