use std::ffi::CStr;
use std::fs;
use std::os::raw::c_void;
use std::ptr;

use dg_core::{DataFormat, DataType, DeviceKind, Shape, Tensor};
use dg_runtime::{
    supports_deployment, supports_device, supports_precision, BackendDescriptor, BackendKind,
    BackendOptions, Error, InferBackend, Result, RknnOptions, RuntimeOption, TensorInfo,
};
use tracing::trace;

#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code
)]
mod sys {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// Returns `true` when the real RKNN backend is compiled in.
pub const fn backend_enabled() -> bool {
    true
}

pub struct RknnBackend {
    context: Option<sys::rknn_context>,
    options: RknnOptions,
    input_infos: Vec<TensorInfo>,
    output_infos: Vec<TensorInfo>,
    input_attrs: Vec<sys::rknn_tensor_attr>,
    output_attrs: Vec<sys::rknn_tensor_attr>,
}

#[derive(Debug)]
struct ModelBuffer {
    data: Vec<u8>,
}

impl ModelBuffer {
    fn new(data: Vec<u8>) -> Result<Self> {
        let _size: u32 = data
            .len()
            .try_into()
            .map_err(|_| Error::InvalidOption("rknn model is too large".to_string()))?;
        Ok(Self { data })
    }

    fn as_ptr(&self) -> *mut c_void {
        self.data.as_ptr() as *mut c_void
    }

    fn len(&self) -> u32 {
        self.data.len() as u32
    }
}

impl RknnBackend {
    fn new() -> Self {
        Self {
            context: None,
            options: RknnOptions::default(),
            input_infos: Vec::new(),
            output_infos: Vec::new(),
            input_attrs: Vec::new(),
            output_attrs: Vec::new(),
        }
    }

    fn context(&self) -> Result<sys::rknn_context> {
        self.context
            .ok_or_else(|| Error::BackendUnavailable("rknn context not initialized".to_string()))
    }

    fn validate_precision(option: &RuntimeOption) -> Result<()> {
        let Some(precision) = option.precision else {
            return Ok(());
        };
        if supports_precision(BackendKind::Rknn, precision) {
            Ok(())
        } else {
            Err(Error::UnsupportedPrecision(precision))
        }
    }

    fn load_model(&self, source: &dg_runtime::ModelSource) -> Result<ModelBuffer> {
        match source {
            dg_runtime::ModelSource::File(path) => {
                let data = fs::read(path).map_err(|err| {
                    Error::BackendUnavailable(format!(
                        "failed to read RKNN model file {}: {err}",
                        path.display()
                    ))
                })?;
                ModelBuffer::new(data)
            }
            dg_runtime::ModelSource::Bytes(bytes) => ModelBuffer::new(bytes.clone()),
        }
    }

    fn init_context(&mut self, option: &RuntimeOption) -> Result<()> {
        trace!("initializing RKNN backend");
        Self::validate_precision(option)?;
        if let Some(device) = option.device
            && !supports_device(BackendKind::Rknn, device)
        {
            return Err(Error::UnsupportedDevice(device));
        }
        if let Some(deploy_mode) = option.deploy_mode
            && !supports_deployment(BackendKind::Rknn, deploy_mode)
        {
            return Err(Error::UnsupportedDeployment(deploy_mode));
        }
        let BackendOptions::Rknn(options) = &option.backend_options else {
            return Err(Error::InvalidOption(
                "rknn backend requires Rknn backend options".to_string(),
            ));
        };
        self.options = options.clone();

        let model = self.load_model(&option.model_source)?;
        let model_ptr = model.as_ptr();
        let model_size = model.len();
        let mut context: sys::rknn_context = Default::default();

        let status =
            unsafe { sys::rknn_init(&mut context, model_ptr, model_size, 0, ptr::null_mut()) };
        check_status(status, "rknn_init")?;
        drop(model);

        if let Some(mask) = self.options.core_mask {
            let mask = map_core_mask(mask)?;
            let status = unsafe { sys::rknn_set_core_mask(context, mask) };
            check_status(status, "rknn_set_core_mask")?;
        }

        self.context = Some(context);
        self.refresh_io_info()?;
        Ok(())
    }

    fn refresh_io_info(&mut self) -> Result<()> {
        let context = self.context()?;
        let mut io_num: sys::rknn_input_output_num = unsafe { std::mem::zeroed() };
        let status = unsafe {
            sys::rknn_query(
                context,
                sys::rknn_query_cmd::RKNN_QUERY_IN_OUT_NUM,
                &mut io_num as *mut _ as *mut c_void,
                std::mem::size_of::<sys::rknn_input_output_num>()
                    .try_into()
                    .expect("rknn_input_output_num size fits in u32"),
            )
        };
        check_status(status, "rknn_query(IN_OUT_NUM)")?;

        self.input_attrs.clear();
        self.output_attrs.clear();
        self.input_infos.clear();
        self.output_infos.clear();

        for index in 0..io_num.n_input {
            let mut attr: sys::rknn_tensor_attr = unsafe { std::mem::zeroed() };
            attr.index = index;
            let status = unsafe {
                sys::rknn_query(
                    context,
                    sys::rknn_query_cmd::RKNN_QUERY_INPUT_ATTR,
                    &mut attr as *mut _ as *mut c_void,
                    std::mem::size_of::<sys::rknn_tensor_attr>()
                        .try_into()
                        .expect("rknn_tensor_attr size fits in u32"),
                )
            };
            check_status(status, "rknn_query(INPUT_ATTR)")?;
            self.input_infos.push(tensor_info_from_attr(&attr)?);
            self.input_attrs.push(attr);
        }

        for index in 0..io_num.n_output {
            let mut attr: sys::rknn_tensor_attr = unsafe { std::mem::zeroed() };
            attr.index = index;
            let status = unsafe {
                sys::rknn_query(
                    context,
                    sys::rknn_query_cmd::RKNN_QUERY_OUTPUT_ATTR,
                    &mut attr as *mut _ as *mut c_void,
                    std::mem::size_of::<sys::rknn_tensor_attr>()
                        .try_into()
                        .expect("rknn_tensor_attr size fits in u32"),
                )
            };
            check_status(status, "rknn_query(OUTPUT_ATTR)")?;
            self.output_infos.push(tensor_info_from_attr(&attr)?);
            self.output_attrs.push(attr);
        }

        Ok(())
    }

    fn set_input_shapes(&mut self, input_shapes: &[Shape]) -> Result<()> {
        if input_shapes.len() != self.input_attrs.len() {
            return Err(Error::InvalidOption(format!(
                "rknn reshape expected {} input shapes, got {}",
                self.input_attrs.len(),
                input_shapes.len()
            )));
        }

        let context = self.context()?;
        let mut attrs = self.input_attrs.clone();
        for (attr, shape) in attrs.iter_mut().zip(input_shapes.iter()) {
            if !self.options.dynamic_shape && shape != &shape_from_attr(attr) {
                return Err(Error::InvalidOption(
                    "rknn model does not permit dynamic reshape".to_string(),
                ));
            }
            update_attr_shape(attr, shape)?;
        }

        let status =
            unsafe { sys::rknn_set_input_shapes(context, attrs.len() as u32, attrs.as_mut_ptr()) };
        check_status(status, "rknn_set_input_shapes")?;
        self.input_attrs = attrs;
        self.refresh_io_info()?;
        Ok(())
    }
}

impl InferBackend for RknnBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Rknn
    }

    fn init(&mut self, option: &RuntimeOption) -> Result<()> {
        let BackendOptions::Rknn(_) = &option.backend_options else {
            return Err(Error::InvalidOption(
                "rknn backend requires Rknn backend options".to_string(),
            ));
        };
        self.init_context(option)
    }

    fn reshape(&mut self, input_shapes: &[Shape]) -> Result<()> {
        self.set_input_shapes(input_shapes)
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
        let context = self.context()?;
        if inputs.len() != self.input_infos.len() {
            return Err(Error::InvalidOption(format!(
                "rknn run expected {} inputs, got {}",
                self.input_infos.len(),
                inputs.len()
            )));
        }

        let input_buffers: Vec<Vec<u8>> = inputs
            .iter()
            .map(|tensor| tensor.buffer().read_bytes())
            .collect();
        let mut inputs_set = Vec::with_capacity(input_buffers.len());
        for (index, buffer) in input_buffers.iter().enumerate() {
            inputs_set.push(sys::rknn_input {
                index: index as u32,
                buf: buffer.as_ptr() as *mut c_void,
                size: buffer
                    .len()
                    .try_into()
                    .map_err(|_| Error::InvalidOption("input buffer too large".to_string()))?,
                pass_through: 1,
                type_: unsafe { std::mem::zeroed() },
                fmt: unsafe { std::mem::zeroed() },
            });
        }

        let status = unsafe {
            sys::rknn_inputs_set(context, inputs_set.len() as u32, inputs_set.as_mut_ptr())
        };
        check_status(status, "rknn_inputs_set")?;

        let status = unsafe { sys::rknn_run(context, ptr::null_mut()) };
        check_status(status, "rknn_run")?;

        let mut outputs = Vec::with_capacity(self.output_infos.len());
        for index in 0..self.output_infos.len() {
            outputs.push(sys::rknn_output {
                want_float: 0,
                is_prealloc: 0,
                index: index as u32,
                buf: ptr::null_mut(),
                size: 0,
            });
        }

        let status = unsafe {
            sys::rknn_outputs_get(
                context,
                outputs.len() as u32,
                outputs.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        if let Err(err) = check_status(status, "rknn_outputs_get") {
            return Err(err);
        }

        let device = dg_core::CpuDevice::new();
        let mut tensors = Vec::with_capacity(outputs.len());
        for (index, output) in outputs.iter().enumerate() {
            if output.buf.is_null() {
                release_outputs(context, &mut outputs)?;
                return Err(Error::Backend(format!(
                    "rknn output {index} returned a null buffer"
                )));
            }
            let info = &self.output_infos[index];
            let tensor = info.allocate(&device)?;
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    output.buf as *const u8,
                    usize::try_from(output.size).expect("output size fits usize"),
                )
            };
            if tensor.buffer().len() != bytes.len() {
                release_outputs(context, &mut outputs)?;
                return Err(Error::Backend(format!(
                    "rknn output size mismatch: expected {}, got {}",
                    tensor.buffer().len(),
                    bytes.len()
                )));
            }
            tensor.buffer().write_from_slice(bytes)?;
            tensors.push(tensor);
        }

        release_outputs(context, &mut outputs)?;
        Ok(tensors)
    }
}

impl Drop for RknnBackend {
    fn drop(&mut self) {
        if let Some(context) = self.context.take() {
            unsafe {
                let _ = sys::rknn_destroy(context);
            }
        }
    }
}

fn create_rknn_backend() -> Box<dyn InferBackend> {
    Box::new(RknnBackend::new())
}

inventory::submit! {
    BackendDescriptor {
        kind: BackendKind::Rknn,
        name: "rknn",
        create: create_rknn_backend,
    }
}

fn check_status(status: i32, context: &str) -> Result<()> {
    if status >= 0 {
        Ok(())
    } else {
        Err(Error::Backend(format!(
            "{context} failed with code {status}"
        )))
    }
}

fn map_core_mask(mask: u32) -> Result<sys::rknn_core_mask> {
    use sys::rknn_core_mask::*;
    match mask {
        0 => Ok(RKNN_NPU_CORE_AUTO),
        1 => Ok(RKNN_NPU_CORE_0),
        2 => Ok(RKNN_NPU_CORE_1),
        4 => Ok(RKNN_NPU_CORE_2),
        3 => Ok(RKNN_NPU_CORE_0_1),
        7 => Ok(RKNN_NPU_CORE_0_1_2),
        0xffff => Ok(RKNN_NPU_CORE_ALL),
        _ => Err(Error::InvalidOption(format!(
            "unsupported RKNN core mask: {mask:#x}"
        ))),
    }
}

fn tensor_info_from_attr(attr: &sys::rknn_tensor_attr) -> Result<TensorInfo> {
    let shape = shape_from_attr(attr);
    let dtype = dtype_from_rknn(attr.type_)?;
    let mut info = TensorInfo::new(shape, dtype);
    if !attr.name.iter().all(|&byte| byte == 0) {
        let name = unsafe { CStr::from_ptr(attr.name.as_ptr()) };
        if let Ok(name) = name.to_str() {
            if !name.is_empty() {
                info = info.with_name(name.to_string());
            }
        }
    }
    if let Some(layout) = layout_from_rknn(attr.fmt) {
        info = info.with_layout(layout);
    }
    Ok(info)
}

fn shape_from_attr(attr: &sys::rknn_tensor_attr) -> Shape {
    let dims = attr
        .dims
        .iter()
        .copied()
        .take(attr.n_dims as usize)
        .map(|dim| dim as usize)
        .collect::<Vec<_>>();
    Shape::new(dims)
}

fn update_attr_shape(attr: &mut sys::rknn_tensor_attr, shape: &Shape) -> Result<()> {
    if shape.rank() > attr.dims.len() {
        return Err(Error::InvalidOption(format!(
            "shape rank {} exceeds RKNN max dims {}",
            shape.rank(),
            attr.dims.len()
        )));
    }
    attr.n_dims = shape.rank() as u32;
    for slot in &mut attr.dims {
        *slot = 0;
    }
    for (slot, dim) in attr.dims.iter_mut().zip(shape.dims().iter()) {
        *slot = (*dim)
            .try_into()
            .map_err(|_| Error::InvalidOption("shape dimension overflows RKNN attr".to_string()))?;
    }
    attr.n_elems = shape
        .element_count()?
        .try_into()
        .map_err(|_| Error::InvalidOption("shape element count overflows RKNN attr".to_string()))?;
    attr.size = attr
        .n_elems
        .checked_mul(bytes_per_element_for_rknn(attr.type_)? as u32)
        .ok_or_else(|| Error::InvalidOption("shape byte size overflow".to_string()))?;
    Ok(())
}

fn bytes_per_element_for_rknn(dtype: sys::rknn_tensor_type) -> Result<usize> {
    match dtype {
        sys::rknn_tensor_type::RKNN_TENSOR_FLOAT32 => Ok(DataType::F32.bytes_per_element_ceil()),
        sys::rknn_tensor_type::RKNN_TENSOR_FLOAT16 => Ok(DataType::F16.bytes_per_element_ceil()),
        sys::rknn_tensor_type::RKNN_TENSOR_INT8 => Ok(DataType::I8.bytes_per_element_ceil()),
        sys::rknn_tensor_type::RKNN_TENSOR_UINT8 => Ok(DataType::U8.bytes_per_element_ceil()),
        sys::rknn_tensor_type::RKNN_TENSOR_INT16 => Ok(DataType::I16.bytes_per_element_ceil()),
        sys::rknn_tensor_type::RKNN_TENSOR_UINT16 => Ok(DataType::U16.bytes_per_element_ceil()),
        sys::rknn_tensor_type::RKNN_TENSOR_INT32 => {
            Ok(DataType::new(dg_core::TypeCode::Int, 32, 1).bytes_per_element_ceil())
        }
        sys::rknn_tensor_type::RKNN_TENSOR_UINT32 => {
            Ok(DataType::new(dg_core::TypeCode::Uint, 32, 1).bytes_per_element_ceil())
        }
        _ => Err(Error::InvalidOption(format!(
            "unsupported RKNN tensor type: {:?}",
            dtype
        ))),
    }
}

fn dtype_from_rknn(dtype: sys::rknn_tensor_type) -> Result<DataType> {
    match dtype {
        sys::rknn_tensor_type::RKNN_TENSOR_FLOAT32 => Ok(DataType::F32),
        sys::rknn_tensor_type::RKNN_TENSOR_FLOAT16 => Ok(DataType::F16),
        sys::rknn_tensor_type::RKNN_TENSOR_INT8 => Ok(DataType::I8),
        sys::rknn_tensor_type::RKNN_TENSOR_UINT8 => Ok(DataType::U8),
        sys::rknn_tensor_type::RKNN_TENSOR_INT16 => Ok(DataType::I16),
        sys::rknn_tensor_type::RKNN_TENSOR_UINT16 => Ok(DataType::U16),
        sys::rknn_tensor_type::RKNN_TENSOR_INT32 => {
            Ok(DataType::new(dg_core::TypeCode::Int, 32, 1))
        }
        sys::rknn_tensor_type::RKNN_TENSOR_UINT32 => {
            Ok(DataType::new(dg_core::TypeCode::Uint, 32, 1))
        }
        _ => Err(Error::InvalidOption(format!(
            "unsupported RKNN tensor type: {:?}",
            dtype
        ))),
    }
}

fn layout_from_rknn(fmt: sys::rknn_tensor_format) -> Option<DataFormat> {
    match fmt {
        sys::rknn_tensor_format::RKNN_TENSOR_NCHW => Some(DataFormat::NCHW),
        sys::rknn_tensor_format::RKNN_TENSOR_NHWC => Some(DataFormat::NHWC),
        _ => None,
    }
}

fn release_outputs(context: sys::rknn_context, outputs: &mut [sys::rknn_output]) -> Result<()> {
    let status =
        unsafe { sys::rknn_outputs_release(context, outputs.len() as u32, outputs.as_mut_ptr()) };
    check_status(status, "rknn_outputs_release")
}
