//! In-process mock of the TensorRT C shim used for deterministic
//! no-hardware tests. Function signatures mirror the bindgen output for
//! `trt_shim.h` so `backend.rs` compiles unchanged against either.
//!
//! The mock deserializes a tiny engine description (see [`encode_mock_engine`])
//! instead of a real TensorRT plan and implements inference as a byte copy
//! from the first input binding into every output binding.

#![allow(non_camel_case_types, clippy::missing_safety_doc)]

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr;
use std::sync::Mutex;

pub const MOCK_ENGINE_MAGIC: &[u8; 10] = b"DGTRTMOCK1";

#[derive(Clone)]
struct MockIo {
    name: CString,
    is_input: bool,
    dtype: i32,
    dims: Vec<i64>,
}

pub struct trt_runtime_handle {
    _private: u8,
}

pub struct trt_engine_handle {
    ios: Vec<MockIo>,
}

pub struct trt_context_handle {
    ios: Vec<MockIo>,
    shapes: HashMap<Vec<u8>, Vec<i64>>,
    addresses: HashMap<Vec<u8>, *mut c_void>,
}

struct MockStream {
    _private: u8,
}

static ALLOCATIONS: Mutex<Option<HashMap<usize, Layout>>> = Mutex::new(None);

/// Serializes a mock engine description understood by
/// [`trt_runtime_deserialize_engine`].
pub fn encode_mock_engine(ios: &[(&str, bool, i32, &[i64])]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(MOCK_ENGINE_MAGIC);
    data.extend_from_slice(&u32::try_from(ios.len()).expect("io count").to_le_bytes());
    for (name, is_input, dtype, dims) in ios {
        data.push(u8::from(*is_input));
        data.extend_from_slice(&dtype.to_le_bytes());
        data.extend_from_slice(&u32::try_from(dims.len()).expect("rank").to_le_bytes());
        for dim in *dims {
            data.extend_from_slice(&dim.to_le_bytes());
        }
        let name_bytes = name.as_bytes();
        data.extend_from_slice(&u32::try_from(name_bytes.len()).expect("name").to_le_bytes());
        data.extend_from_slice(name_bytes);
    }
    data
}

fn decode_mock_engine(data: &[u8]) -> Option<Vec<MockIo>> {
    let mut cursor = data;
    let magic = take(&mut cursor, MOCK_ENGINE_MAGIC.len())?;
    if magic != MOCK_ENGINE_MAGIC {
        return None;
    }
    let count = u32::from_le_bytes(take(&mut cursor, 4)?.try_into().ok()?) as usize;
    let mut ios = Vec::with_capacity(count);
    for _ in 0..count {
        let is_input = take(&mut cursor, 1)?[0] != 0;
        let dtype = i32::from_le_bytes(take(&mut cursor, 4)?.try_into().ok()?);
        let rank = u32::from_le_bytes(take(&mut cursor, 4)?.try_into().ok()?) as usize;
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            dims.push(i64::from_le_bytes(take(&mut cursor, 8)?.try_into().ok()?));
        }
        let name_len = u32::from_le_bytes(take(&mut cursor, 4)?.try_into().ok()?) as usize;
        let name = CString::new(take(&mut cursor, name_len)?).ok()?;
        ios.push(MockIo {
            name,
            is_input,
            dtype,
            dims,
        });
    }
    Some(ios)
}

fn take<'a>(cursor: &mut &'a [u8], len: usize) -> Option<&'a [u8]> {
    if cursor.len() < len {
        return None;
    }
    let (head, tail) = cursor.split_at(len);
    *cursor = tail;
    Some(head)
}

fn element_count(dims: &[i64]) -> Option<usize> {
    dims.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(usize::try_from(*dim).ok()?)
    })
}

fn dtype_bytes(dtype: i32) -> usize {
    match dtype {
        crate::convert::TRT_DTYPE_FLOAT => 4,
        crate::convert::TRT_DTYPE_INT32 => 4,
        crate::convert::TRT_DTYPE_HALF | crate::convert::TRT_DTYPE_BF16 => 2,
        crate::convert::TRT_DTYPE_INT64 => 8,
        _ => 1,
    }
}

pub unsafe fn trt_runtime_create() -> *mut trt_runtime_handle {
    Box::into_raw(Box::new(trt_runtime_handle { _private: 0 }))
}

pub unsafe fn trt_runtime_destroy(handle: *mut trt_runtime_handle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

pub unsafe fn trt_runtime_deserialize_engine(
    runtime: *mut trt_runtime_handle,
    data: *const c_void,
    size: usize,
) -> *mut trt_engine_handle {
    if runtime.is_null() || data.is_null() || size == 0 {
        return ptr::null_mut();
    }
    let bytes = std::slice::from_raw_parts(data as *const u8, size);
    match decode_mock_engine(bytes) {
        Some(ios) => Box::into_raw(Box::new(trt_engine_handle { ios })),
        None => ptr::null_mut(),
    }
}

pub unsafe fn trt_engine_destroy(handle: *mut trt_engine_handle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

pub unsafe fn trt_engine_create_context(handle: *mut trt_engine_handle) -> *mut trt_context_handle {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let ios = (*handle).ios.clone();
    let shapes = ios
        .iter()
        .map(|io| (io.name.as_bytes().to_vec(), io.dims.clone()))
        .collect();
    Box::into_raw(Box::new(trt_context_handle {
        ios,
        shapes,
        addresses: HashMap::new(),
    }))
}

pub unsafe fn trt_context_destroy(handle: *mut trt_context_handle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

pub unsafe fn trt_engine_num_io(handle: *mut trt_engine_handle) -> c_int {
    if handle.is_null() {
        return -1;
    }
    let engine = &*handle;
    c_int::try_from(engine.ios.len()).unwrap_or(-1)
}

pub unsafe fn trt_engine_io_name(handle: *mut trt_engine_handle, index: c_int) -> *const c_char {
    if handle.is_null() {
        return ptr::null();
    }
    let Ok(index) = usize::try_from(index) else {
        return ptr::null();
    };
    let engine = &*handle;
    match engine.ios.get(index) {
        Some(io) => io.name.as_ptr(),
        None => ptr::null(),
    }
}

pub unsafe fn trt_engine_io_is_input(handle: *mut trt_engine_handle, index: c_int) -> c_int {
    if handle.is_null() {
        return 0;
    }
    let engine = &*handle;
    usize::try_from(index)
        .ok()
        .and_then(|index| engine.ios.get(index))
        .is_some_and(|io| io.is_input)
        .into()
}

pub unsafe fn trt_engine_io_dtype(handle: *mut trt_engine_handle, index: c_int) -> c_int {
    if handle.is_null() {
        return -1;
    }
    let engine = &*handle;
    usize::try_from(index)
        .ok()
        .and_then(|index| engine.ios.get(index))
        .map_or(-1, |io| io.dtype)
}

pub unsafe fn trt_engine_io_shape(
    handle: *mut trt_engine_handle,
    index: c_int,
    dims: *mut i64,
    max_rank: usize,
) -> c_int {
    if handle.is_null() || dims.is_null() {
        return -1;
    }
    let engine = &*handle;
    let Some(io) = usize::try_from(index)
        .ok()
        .and_then(|index| engine.ios.get(index))
    else {
        return -1;
    };
    if io.dims.len() > max_rank {
        return -1;
    }
    ptr::copy_nonoverlapping(io.dims.as_ptr(), dims, io.dims.len());
    c_int::try_from(io.dims.len()).unwrap_or(-1)
}

pub unsafe fn trt_context_set_input_shape(
    handle: *mut trt_context_handle,
    name: *const c_char,
    dims: *const i64,
    rank: usize,
) -> c_int {
    if handle.is_null() || name.is_null() || dims.is_null() {
        return 0;
    }
    let name = CStr::from_ptr(name).to_bytes().to_vec();
    let requested = std::slice::from_raw_parts(dims, rank);
    let context = &mut *handle;
    let Some(io) = context
        .ios
        .iter()
        .find(|io| io.is_input && io.name.as_bytes() == name)
    else {
        return 0;
    };
    if io.dims.len() != rank {
        return 0;
    }
    let compatible = io
        .dims
        .iter()
        .zip(requested)
        .all(|(engine, req)| *engine == -1 || engine == req);
    if !compatible {
        return 0;
    }
    context.shapes.insert(name, requested.to_vec());
    // Emulate TensorRT shape propagation: resolve dynamic output dims from
    // the input shape when ranks line up.
    let output_names: Vec<Vec<u8>> = context
        .ios
        .iter()
        .filter(|io| !io.is_input)
        .map(|io| io.name.as_bytes().to_vec())
        .collect();
    for output_name in output_names {
        if let Some(shape) = context.shapes.get_mut(&output_name) {
            if shape.len() == rank {
                for (dim, resolved) in shape.iter_mut().zip(requested) {
                    if *dim == -1 {
                        *dim = *resolved;
                    }
                }
            }
        }
    }
    1
}

pub unsafe fn trt_context_get_tensor_shape(
    handle: *mut trt_context_handle,
    name: *const c_char,
    dims: *mut i64,
    max_rank: usize,
) -> c_int {
    if handle.is_null() || name.is_null() || dims.is_null() {
        return -1;
    }
    let name = CStr::from_ptr(name).to_bytes();
    let context = &*handle;
    let Some(shape) = context.shapes.get(name) else {
        return -1;
    };
    if shape.len() > max_rank {
        return -1;
    }
    ptr::copy_nonoverlapping(shape.as_ptr(), dims, shape.len());
    c_int::try_from(shape.len()).unwrap_or(-1)
}

pub unsafe fn trt_context_set_tensor_address(
    handle: *mut trt_context_handle,
    name: *const c_char,
    address: *mut c_void,
) -> c_int {
    if handle.is_null() || name.is_null() || address.is_null() {
        return 0;
    }
    let name = CStr::from_ptr(name).to_bytes().to_vec();
    let context = &mut *handle;
    if !context.ios.iter().any(|io| io.name.as_bytes() == name) {
        return 0;
    }
    context.addresses.insert(name, address);
    1
}

pub unsafe fn trt_context_enqueue(handle: *mut trt_context_handle, _stream: *mut c_void) -> c_int {
    if handle.is_null() {
        return 0;
    }
    let context = &*handle;
    let Some(input) = context.ios.iter().find(|io| io.is_input) else {
        return 0;
    };
    let Some(input_addr) = context.addresses.get(input.name.as_bytes()) else {
        return 0;
    };
    let Some(input_shape) = context.shapes.get(input.name.as_bytes()) else {
        return 0;
    };
    let Some(input_elements) = element_count(input_shape) else {
        return 0;
    };
    let input_bytes = input_elements * dtype_bytes(input.dtype);
    for output in context.ios.iter().filter(|io| !io.is_input) {
        let Some(output_addr) = context.addresses.get(output.name.as_bytes()) else {
            return 0;
        };
        let Some(output_shape) = context.shapes.get(output.name.as_bytes()) else {
            return 0;
        };
        let Some(output_elements) = element_count(output_shape) else {
            return 0;
        };
        let output_bytes = output_elements * dtype_bytes(output.dtype);
        let copied = input_bytes.min(output_bytes);
        ptr::copy_nonoverlapping(*input_addr as *const u8, *output_addr as *mut u8, copied);
    }
    1
}

pub unsafe fn trt_cuda_device_count() -> c_int {
    1
}

pub unsafe fn trt_cuda_set_device(device: c_int) -> c_int {
    c_int::from(device == 0)
}

pub unsafe fn trt_cuda_stream_create() -> *mut c_void {
    Box::into_raw(Box::new(MockStream { _private: 0 })) as *mut c_void
}

pub unsafe fn trt_cuda_stream_destroy(stream: *mut c_void) {
    if !stream.is_null() {
        drop(Box::from_raw(stream as *mut MockStream));
    }
}

pub unsafe fn trt_cuda_stream_synchronize(stream: *mut c_void) -> c_int {
    c_int::from(!stream.is_null())
}

pub unsafe fn trt_cuda_malloc(size: usize) -> *mut c_void {
    let Ok(layout) = Layout::from_size_align(size.max(1), 64) else {
        return ptr::null_mut();
    };
    let allocation = alloc_zeroed(layout);
    if allocation.is_null() {
        return ptr::null_mut();
    }
    let mut allocations = ALLOCATIONS.lock().expect("mock allocation registry");
    allocations
        .get_or_insert_with(HashMap::new)
        .insert(allocation as usize, layout);
    allocation as *mut c_void
}

pub unsafe fn trt_cuda_free(pointer: *mut c_void) {
    if pointer.is_null() {
        return;
    }
    let layout = {
        let mut allocations = ALLOCATIONS.lock().expect("mock allocation registry");
        allocations
            .get_or_insert_with(HashMap::new)
            .remove(&(pointer as usize))
    };
    if let Some(layout) = layout {
        dealloc(pointer as *mut u8, layout);
    }
}

pub unsafe fn trt_cuda_memcpy_h2d(dst: *mut c_void, src: *const c_void, size: usize) -> c_int {
    if dst.is_null() || src.is_null() {
        return 0;
    }
    ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, size);
    1
}

pub unsafe fn trt_cuda_memcpy_d2h(dst: *mut c_void, src: *const c_void, size: usize) -> c_int {
    if dst.is_null() || src.is_null() {
        return 0;
    }
    ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, size);
    1
}
