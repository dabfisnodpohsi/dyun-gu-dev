//! SDK-free RKNN ABI shim used by the backend adapter tests.

#![allow(
    dead_code,
    non_camel_case_types,
    clippy::derivable_impls,
    clippy::missing_safety_doc
)]

use std::ffi::{c_char, c_void};
use std::ptr;

pub const MOCK_MODEL_MAGIC: &[u8; 10] = b"DRKNNMOCK1";
const MAX_DIMS: usize = 16;
const NAME_LEN: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum rknn_tensor_type {
    #[default]
    RKNN_TENSOR_FLOAT32 = 0,
    RKNN_TENSOR_FLOAT16 = 1,
    RKNN_TENSOR_INT8 = 2,
    RKNN_TENSOR_UINT8 = 3,
    RKNN_TENSOR_INT16 = 4,
    RKNN_TENSOR_UINT16 = 5,
    RKNN_TENSOR_INT32 = 6,
    RKNN_TENSOR_UINT32 = 7,
    RKNN_TENSOR_UNKNOWN = 255,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum rknn_tensor_format {
    RKNN_TENSOR_NCHW = 0,
    #[default]
    RKNN_TENSOR_NHWC = 1,
    RKNN_TENSOR_FORMAT_UNKNOWN = 255,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum rknn_tensor_qnt_type {
    #[default]
    RKNN_TENSOR_QNT_NONE = 0,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rknn_query_cmd {
    RKNN_QUERY_IN_OUT_NUM = 0,
    RKNN_QUERY_INPUT_ATTR = 1,
    RKNN_QUERY_OUTPUT_ATTR = 2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rknn_core_mask {
    RKNN_NPU_CORE_AUTO = 0,
    RKNN_NPU_CORE_0 = 1,
    RKNN_NPU_CORE_1 = 2,
    RKNN_NPU_CORE_2 = 4,
    RKNN_NPU_CORE_0_1 = 3,
    RKNN_NPU_CORE_0_1_2 = 7,
    RKNN_NPU_CORE_ALL = 0xffff,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct rknn_input_output_num {
    pub n_input: u32,
    pub n_output: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct rknn_tensor_attr {
    pub index: u32,
    pub n_dims: u32,
    pub dims: [u32; MAX_DIMS],
    pub n_elems: u32,
    pub size: u32,
    pub fmt: rknn_tensor_format,
    pub type_: rknn_tensor_type,
    pub qnt_type: rknn_tensor_qnt_type,
    pub fl: i8,
    pub zp: i32,
    pub scale: f32,
    pub w_stride: u32,
    pub size_with_stride: u32,
    pub name: [c_char; NAME_LEN],
}

impl Default for rknn_tensor_attr {
    fn default() -> Self {
        Self {
            index: 0,
            n_dims: 0,
            dims: [0; MAX_DIMS],
            n_elems: 0,
            size: 0,
            fmt: rknn_tensor_format::default(),
            type_: rknn_tensor_type::default(),
            qnt_type: rknn_tensor_qnt_type::default(),
            fl: 0,
            zp: 0,
            scale: 1.0,
            w_stride: 0,
            size_with_stride: 0,
            name: [0; NAME_LEN],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct rknn_tensor_mem {
    pub virt_addr: *mut c_void,
    pub phys_addr: u64,
    pub size: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct rknn_input {
    pub index: u32,
    pub buf: *mut c_void,
    pub size: u32,
    pub pass_through: u8,
    pub type_: rknn_tensor_type,
    pub fmt: rknn_tensor_format,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct rknn_output {
    pub want_float: u8,
    pub is_prealloc: u8,
    pub index: u32,
    pub buf: *mut c_void,
    pub size: u32,
}

pub type rknn_context = usize;

struct MockContext {
    input: Vec<u8>,
    direct_input: *const u8,
    direct_output: *mut u8,
    input_attr: rknn_tensor_attr,
    output_attr: rknn_tensor_attr,
}

pub fn encode_mock_model() -> Vec<u8> {
    MOCK_MODEL_MAGIC.to_vec()
}

fn attr(name: &[u8]) -> rknn_tensor_attr {
    let mut attr = rknn_tensor_attr {
        n_dims: 2,
        dims: [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        n_elems: 4,
        size: 16,
        w_stride: 4,
        ..rknn_tensor_attr::default()
    };
    for (slot, byte) in attr.name.iter_mut().zip(name.iter().copied()) {
        *slot = i8::try_from(byte).unwrap_or_default();
    }
    attr
}

pub unsafe fn rknn_init(
    context: *mut rknn_context,
    model: *mut c_void,
    size: u32,
    _flag: u32,
    _extend: *mut c_void,
) -> i32 {
    if context.is_null() || model.is_null() || size < MOCK_MODEL_MAGIC.len() as u32 {
        return -1;
    }
    let bytes = std::slice::from_raw_parts(model.cast::<u8>(), size as usize);
    if !bytes.starts_with(MOCK_MODEL_MAGIC) {
        return -1;
    }
    let ctx = Box::new(MockContext {
        input: vec![0; 16],
        direct_input: ptr::null(),
        direct_output: ptr::null_mut(),
        input_attr: attr(b"input"),
        output_attr: attr(b"output"),
    });
    *context = Box::into_raw(ctx) as usize;
    0
}

pub unsafe fn rknn_destroy(context: rknn_context) -> i32 {
    if context != 0 {
        drop(Box::from_raw(context as *mut MockContext));
    }
    0
}

pub unsafe fn rknn_set_core_mask(_context: rknn_context, _mask: rknn_core_mask) -> i32 {
    0
}

pub unsafe fn rknn_query(
    context: rknn_context,
    command: rknn_query_cmd,
    value: *mut c_void,
    _size: u32,
) -> i32 {
    if context == 0 || value.is_null() {
        return -1;
    }
    let context = &*(context as *mut MockContext);
    match command {
        rknn_query_cmd::RKNN_QUERY_IN_OUT_NUM => {
            ptr::write(
                value.cast(),
                rknn_input_output_num {
                    n_input: 1,
                    n_output: 1,
                },
            );
        }
        rknn_query_cmd::RKNN_QUERY_INPUT_ATTR => {
            ptr::write(value.cast(), context.input_attr);
        }
        rknn_query_cmd::RKNN_QUERY_OUTPUT_ATTR => {
            ptr::write(value.cast(), context.output_attr);
        }
    }
    0
}

pub unsafe fn rknn_set_input_shapes(
    context: rknn_context,
    count: u32,
    attrs: *mut rknn_tensor_attr,
) -> i32 {
    if context == 0 || count != 1 || attrs.is_null() {
        return -1;
    }
    (*(context as *mut MockContext)).input_attr = *attrs;
    (*(context as *mut MockContext)).output_attr = *attrs;
    0
}

pub unsafe fn rknn_create_mem(_context: rknn_context, _size: u32) -> *mut rknn_tensor_mem {
    let size = usize::try_from(_size).unwrap_or_default();
    if size == 0 {
        return ptr::null_mut();
    }
    let mut bytes = vec![0u8; size];
    let virt_addr = bytes.as_mut_ptr().cast();
    std::mem::forget(bytes);
    Box::into_raw(Box::new(rknn_tensor_mem {
        virt_addr,
        phys_addr: 0,
        size: _size,
        flags: 0,
    }))
}

pub unsafe fn rknn_create_mem_from_fd(
    _context: rknn_context,
    fd: i32,
    size: u32,
) -> *mut rknn_tensor_mem {
    if fd <= 0 || size == 0 {
        return ptr::null_mut();
    }
    let mut bytes = (0..size).map(|value| value as u8).collect::<Vec<_>>();
    let virt_addr = bytes.as_mut_ptr().cast();
    std::mem::forget(bytes);
    Box::into_raw(Box::new(rknn_tensor_mem {
        virt_addr,
        phys_addr: u64::try_from(fd).unwrap_or_default(),
        size,
        flags: 1,
    }))
}

pub unsafe fn rknn_destroy_mem(_context: rknn_context, mem: *mut rknn_tensor_mem) -> i32 {
    if !mem.is_null() {
        if (*mem).flags != 1 && !(*mem).virt_addr.is_null() {
            drop(Vec::from_raw_parts(
                (*mem).virt_addr.cast::<u8>(),
                (*mem).size as usize,
                (*mem).size as usize,
            ));
        }
        drop(Box::from_raw(mem));
    }
    0
}

pub unsafe fn rknn_set_io_mem(
    context: rknn_context,
    mem: *mut rknn_tensor_mem,
    attr: *mut rknn_tensor_attr,
) -> i32 {
    if context == 0 || mem.is_null() || attr.is_null() {
        return -1;
    }
    let context = &mut *(context as *mut MockContext);
    if (*mem).virt_addr.is_null() {
        return -1;
    }
    if (*mem).flags == 1 || context.direct_input.is_null() {
        context.direct_input = (*mem).virt_addr.cast();
    } else {
        context.direct_output = (*mem).virt_addr.cast();
    }
    0
}

pub unsafe fn rknn_inputs_set(context: rknn_context, count: u32, inputs: *mut rknn_input) -> i32 {
    if context == 0 || count != 1 || inputs.is_null() {
        return -1;
    }
    let input = &*inputs;
    if input.buf.is_null() || input.size != 16 {
        return -1;
    }
    (*(context as *mut MockContext)).input =
        std::slice::from_raw_parts(input.buf.cast::<u8>(), 16).to_vec();
    (*(context as *mut MockContext)).direct_input = ptr::null();
    0
}

pub unsafe fn rknn_run(context: rknn_context, _extend: *mut c_void) -> i32 {
    if context == 0 {
        -1
    } else {
        let context = &mut *(context as *mut MockContext);
        if !context.direct_input.is_null() {
            let len = context.input.len();
            let source = std::slice::from_raw_parts(context.direct_input, len);
            context.input.copy_from_slice(source);
            if !context.direct_output.is_null() {
                std::ptr::copy_nonoverlapping(source.as_ptr(), context.direct_output, len);
            }
        }
        0
    }
}

pub unsafe fn rknn_outputs_get(
    context: rknn_context,
    count: u32,
    outputs: *mut rknn_output,
    _extend: *mut c_void,
) -> i32 {
    if context == 0 || count != 1 || outputs.is_null() {
        return -1;
    }
    let bytes = (*(context as *mut MockContext)).input.clone();
    let ptr = bytes.as_ptr() as *mut u8;
    std::mem::forget(bytes);
    (*outputs).buf = ptr.cast();
    (*outputs).size = 16;
    0
}

pub unsafe fn rknn_outputs_release(
    _context: rknn_context,
    count: u32,
    outputs: *mut rknn_output,
) -> i32 {
    if count != 1 || outputs.is_null() {
        return -1;
    }
    let output = &mut *outputs;
    if !output.buf.is_null() {
        drop(Vec::from_raw_parts(
            output.buf.cast::<u8>(),
            output.size as usize,
            output.size as usize,
        ));
        output.buf = ptr::null_mut();
    }
    0
}
