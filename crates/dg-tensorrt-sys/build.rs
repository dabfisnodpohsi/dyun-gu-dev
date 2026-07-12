use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=TENSORRT_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    if env::var_os("CARGO_FEATURE_BACKEND").is_none() {
        return;
    }

    let tensorrt_root = env::var_os("TENSORRT_ROOT")
        .map(PathBuf::from)
        .expect("TENSORRT_ROOT must point to a TensorRT SDK when the backend feature is enabled");
    let include_dir = tensorrt_root.join("include");
    if !include_dir.exists() {
        panic!("TENSORRT_ROOT does not contain an include directory");
    }

    let lib_dir = if tensorrt_root.join("lib").exists() {
        tensorrt_root.join("lib")
    } else if tensorrt_root.join("lib64").exists() {
        tensorrt_root.join("lib64")
    } else {
        panic!("TENSORRT_ROOT does not contain a lib or lib64 directory");
    };

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=nvinfer");
    println!("cargo:rustc-link-lib=dylib=nvonnxparser");
    println!("cargo:rustc-link-lib=dylib=cudart");

    let cuda_root = env::var_os("CUDA_ROOT").map(PathBuf::from);
    let mut cuda_include = None;
    if let Some(cuda_root) = cuda_root {
        let cuda_lib = if cuda_root.join("lib64").exists() {
            cuda_root.join("lib64")
        } else {
            cuda_root.join("lib")
        };
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        let include = cuda_root.join("include");
        if include.exists() {
            cuda_include = Some(include);
        }
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let shim_src = out_dir.join("trt_shim.cpp");
    let shim_hdr = out_dir.join("trt_shim.h");
    fs::write(
        &shim_hdr,
        concat!(
            "#include <cstddef>\n",
            "#include <cstdint>\n",
            "extern \"C\" {\n",
            "struct trt_runtime_handle;\n",
            "struct trt_engine_handle;\n",
            "struct trt_context_handle;\n",
            "trt_runtime_handle* trt_runtime_create();\n",
            "void trt_runtime_destroy(trt_runtime_handle*);\n",
            "trt_engine_handle* trt_runtime_deserialize_engine(trt_runtime_handle*, const void*, size_t);\n",
            "void trt_engine_destroy(trt_engine_handle*);\n",
            "trt_context_handle* trt_engine_create_context(trt_engine_handle*);\n",
            "void trt_context_destroy(trt_context_handle*);\n",
            "int trt_engine_num_io(trt_engine_handle*);\n",
            "const char* trt_engine_io_name(trt_engine_handle*, int);\n",
            "int trt_engine_io_is_input(trt_engine_handle*, int);\n",
            "int trt_engine_io_dtype(trt_engine_handle*, int);\n",
            "int trt_engine_io_shape(trt_engine_handle*, int, int64_t*, size_t);\n",
            "int trt_context_set_input_shape(trt_context_handle*, const char*, const int64_t*, size_t);\n",
            "int trt_context_get_tensor_shape(trt_context_handle*, const char*, int64_t*, size_t);\n",
            "int trt_context_set_tensor_address(trt_context_handle*, const char*, void*);\n",
            "int trt_context_enqueue(trt_context_handle*, void*);\n",
            "int trt_cuda_device_count();\n",
            "int trt_cuda_set_device(int);\n",
            "void* trt_cuda_stream_create();\n",
            "void trt_cuda_stream_destroy(void*);\n",
            "int trt_cuda_stream_synchronize(void*);\n",
            "void* trt_cuda_malloc(size_t);\n",
            "void trt_cuda_free(void*);\n",
            "int trt_cuda_memcpy_h2d(void*, const void*, size_t);\n",
            "int trt_cuda_memcpy_d2h(void*, const void*, size_t);\n",
            "}\n",
        ),
    )
    .expect("write shim header");
    fs::write(
        &shim_src,
        r#"
#include "trt_shim.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

namespace {
class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    (void)severity;
    (void)msg;
  }
};
Logger g_logger;
}

extern "C" {
struct trt_runtime_handle { nvinfer1::IRuntime* ptr; };
struct trt_engine_handle { nvinfer1::ICudaEngine* ptr; };
struct trt_context_handle { nvinfer1::IExecutionContext* ptr; };

trt_runtime_handle* trt_runtime_create() {
  auto* runtime = nvinfer1::createInferRuntime(g_logger);
  if (!runtime) {
    return nullptr;
  }
  return new trt_runtime_handle{runtime};
}

void trt_runtime_destroy(trt_runtime_handle* handle) {
  if (!handle) return;
  delete handle->ptr;
  delete handle;
}

trt_engine_handle* trt_runtime_deserialize_engine(trt_runtime_handle* runtime, const void* data, size_t size) {
  if (!runtime || !runtime->ptr) return nullptr;
  auto* engine = runtime->ptr->deserializeCudaEngine(data, size);
  if (!engine) return nullptr;
  return new trt_engine_handle{engine};
}

void trt_engine_destroy(trt_engine_handle* handle) {
  if (!handle) return;
  delete handle->ptr;
  delete handle;
}

trt_context_handle* trt_engine_create_context(trt_engine_handle* handle) {
  if (!handle || !handle->ptr) return nullptr;
  auto* context = handle->ptr->createExecutionContext();
  if (!context) return nullptr;
  return new trt_context_handle{context};
}

void trt_context_destroy(trt_context_handle* handle) {
  if (!handle) return;
  delete handle->ptr;
  delete handle;
}

int trt_engine_num_io(trt_engine_handle* handle) {
  if (!handle || !handle->ptr) return -1;
  return handle->ptr->getNbIOTensors();
}

const char* trt_engine_io_name(trt_engine_handle* handle, int index) {
  if (!handle || !handle->ptr) return nullptr;
  return handle->ptr->getIOTensorName(index);
}

int trt_engine_io_is_input(trt_engine_handle* handle, int index) {
  if (!handle || !handle->ptr) return 0;
  auto name = handle->ptr->getIOTensorName(index);
  return name && handle->ptr->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
}

int trt_engine_io_dtype(trt_engine_handle* handle, int index) {
  if (!handle || !handle->ptr) return -1;
  auto name = handle->ptr->getIOTensorName(index);
  if (!name) return -1;
  return static_cast<int>(handle->ptr->getTensorDataType(name));
}

int trt_engine_io_shape(trt_engine_handle* handle, int index, int64_t* dims, size_t max_rank) {
  if (!handle || !handle->ptr || !dims) return -1;
  auto name = handle->ptr->getIOTensorName(index);
  if (!name) return -1;
  auto shape = handle->ptr->getTensorShape(name);
  if (shape.nbDims < 0 || static_cast<size_t>(shape.nbDims) > max_rank) return -1;
  for (int i = 0; i < shape.nbDims; ++i) {
    dims[i] = static_cast<int64_t>(shape.d[i]);
  }
  return shape.nbDims;
}

int trt_context_set_input_shape(trt_context_handle* handle, const char* name, const int64_t* dims, size_t rank) {
  if (!handle || !handle->ptr || !name) return 0;
  nvinfer1::Dims shape{};
  shape.nbDims = static_cast<int>(rank);
  for (size_t i = 0; i < rank && i < static_cast<size_t>(nvinfer1::Dims::MAX_DIMS); ++i) {
    shape.d[i] = static_cast<int32_t>(dims[i]);
  }
  return handle->ptr->setInputShape(name, shape) ? 1 : 0;
}

int trt_context_get_tensor_shape(trt_context_handle* handle, const char* name, int64_t* dims, size_t max_rank) {
  if (!handle || !handle->ptr || !name || !dims) return -1;
  auto shape = handle->ptr->getTensorShape(name);
  if (shape.nbDims < 0 || static_cast<size_t>(shape.nbDims) > max_rank) return -1;
  for (int i = 0; i < shape.nbDims; ++i) {
    dims[i] = static_cast<int64_t>(shape.d[i]);
  }
  return shape.nbDims;
}

int trt_context_set_tensor_address(trt_context_handle* handle, const char* name, void* ptr) {
  if (!handle || !handle->ptr || !name) return 0;
  return handle->ptr->setTensorAddress(name, ptr) ? 1 : 0;
}

int trt_context_enqueue(trt_context_handle* handle, void* stream) {
  if (!handle || !handle->ptr) return 0;
  return handle->ptr->enqueueV3(static_cast<cudaStream_t>(stream)) ? 1 : 0;
}

int trt_cuda_device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) return -1;
  return count;
}

int trt_cuda_set_device(int device) {
  return cudaSetDevice(device) == cudaSuccess ? 1 : 0;
}

void* trt_cuda_stream_create() {
  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) return nullptr;
  return static_cast<void*>(stream);
}

void trt_cuda_stream_destroy(void* stream) {
  if (!stream) return;
  cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

int trt_cuda_stream_synchronize(void* stream) {
  return cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) == cudaSuccess ? 1 : 0;
}

void* trt_cuda_malloc(size_t size) {
  void* ptr = nullptr;
  if (cudaMalloc(&ptr, size) != cudaSuccess) return nullptr;
  return ptr;
}

void trt_cuda_free(void* ptr) {
  if (!ptr) return;
  cudaFree(ptr);
}

int trt_cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
  return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) == cudaSuccess ? 1 : 0;
}

int trt_cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
  return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) == cudaSuccess ? 1 : 0;
}
}
"#,
    )
    .expect("write shim source");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file(&shim_src)
        .include(&include_dir)
        .flag_if_supported("-std=c++17");
    if let Some(cuda_include) = cuda_include.as_ref() {
        build.include(cuda_include);
    }
    build.compile("dg_tensorrt_shim");

    let bindings = bindgen::Builder::default()
        .header(shim_hdr.to_string_lossy().into_owned())
        .clang_arg(format!("-I{}", include_dir.display()))
        .derive_default(true)
        .derive_eq(true)
        .allowlist_function("trt_.*")
        .allowlist_type("trt_.*")
        .generate()
        .expect("generate TensorRT bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("write TensorRT bindings");
}
