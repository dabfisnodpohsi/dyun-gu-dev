# dyun-gu-dev：Rust 多芯片推理框架 —— 技术方案与开发计划

> 版本：v0.1（方案草案，待确认）
> 目标：用 Rust 编写一个跨芯片、跨部署模式的流式推理框架，统一支持 **OpenVINO / TensorRT / RKNN2 / Sophon** 四类推理后端，覆盖多种数值精度、多种设备拓扑与负载均衡策略，并集成多媒体编解码与流媒体能力。

---

## 1. 项目目标与范围

### 1.1 核心目标
1. **多推理后端统一**：在同一套 API / 图配置下运行于 OpenVINO（Intel CPU/GPU/NPU）、TensorRT（NVIDIA GPU）、RKNN2（Rockchip NPU）、Sophon（算能 BM 系列）。
2. **多数值精度**：统一张量类型系统支持 `fp32 / fp16 / bf16 / fp8 / fp4 / int16 / uint16 / int8 / uint8 / int4`，并携带量化元信息（scale / zero_point / 量化类型）。
3. **多设备拓扑与负载均衡**：
   - 单芯片；
   - 单卡多核心（核心间负载均衡，如 RK3588 三核 NPU、Sophon 多核）；
   - 多卡多核心；
   - `auto`（运行时自动分配核心/设备）或显式指定核心/设备运行。
4. **SoC 与 Host（PCIe）两种部署模式**：同一套代码，编译期/运行期切换。
5. **流式处理**：借鉴 sophon-stream 的 element/graph/pipeline 模型，用配置驱动构建“解码 → 预处理 → 推理 → 后处理 → 跟踪 → OSD → 编码/推流”的完整流水线。
6. **多媒体与流媒体**：编解码与图像处理依赖 `avcodec-rs`，流媒体拉流/推流依赖 `cheetah-media-server-rs`。
7. **稳定 C ABI**：对外暴露一套 C API（`dg-capi`），方便后续 Python / Go / C++ 等多语言绑定；首期不做具体语言绑定与 serving，但 C API 与头文件作为一等交付物。
8. **广泛交叉编译**：支持 x86_64 / aarch64 / Android / RISC-V 等目标的交叉编译（Host 与 SoC 双模式）。

### 1.2 非目标（首期不做）
- 模型训练与量化工具链（依赖各厂商既有工具，如 rknn-toolkit2、TensorRT trtexec、OpenVINO MO、算能 tpu-mlir）。
- 自研算子/编译器。
- Python 绑定与 serving（gRPC/HTTP）——**首期不做**，但通过 §10.3 的 C ABI 预留接口，后续可独立增量落地。
- Web 管理控制台（可作为后续独立项目）。

---

## 2. 参考项目分析结论

| 项目 | 语言 | 对本项目的价值 | 借鉴点 |
|------|------|----------------|--------|
| **sophgo/sophon-stream** | C++ | 流式插件框架的**图拓扑/线程/背压/配置**范式 | Engine→Graph→Element；Connector+DataPipe 队列；每 element 多线程+多 DataPipe；JSON 图配置；Group 合并 pre/infer/post |
| **ChungTak/nndeploy** | C++ | **后端抽象 + 统一张量/设备/内存 + DAG 执行**的最佳蓝本 | `Inference` 后端基类 + 工厂注册；`Device/Buffer/Tensor` 三层内存抽象；`DataType(code/bits/lanes)` 与 `DataFormat`；`Node/Edge/Graph/Executor` + `ThreadPool`(work-stealing) + `ParallelType(Sequential/Task/Pipeline)` |
| **ChungTak/DyunGuDeploy** | C++ (FastDeploy 体系) | **统一 RuntimeOption + Backend 开关 + serving/streamer** 的产品化组织 | `RuntimeOption` 统一入口；`ENABLE_*_BACKEND` 编译开关；`BaseBackend(Init/Infer/GetInputInfo...)`；`TensorInfo(name/shape/dtype)`；serving/streamer/多语言绑定分层 |
| **RKNN2 (rknpu2 / rknn_model_zoo / doc)** | C | RKNN 后端适配的**API 流程/量化/多核**细节 | `rknn_init/query/inputs_set/run/outputs_get/destroy`；`rknn_tensor_attr(fmt/type/qnt_type/zp/scale/stride)`；`rknn_set_core_mask`（AUTO/0/1/2/0_1/0_1_2/ALL）；zero-copy `rknn_create_mem`/`rknn_set_io_mem`；`rknn_set_input_shapes` 动态 shape |

**关键结论**：架构上以 **nndeploy 的运行时/后端/张量抽象** 为骨架，叠加 **sophon-stream 的流式 element/graph 编排层**，产品化组织参考 **DyunGuDeploy(FastDeploy)** 的 `RuntimeOption` 与后端开关，全部用 Rust 惯用法（trait + 泛型 + RAII + feature flag + `-sys` crate）重写。

---

## 3. 依赖仓库现状（已确认可访问）

两个依赖仓库已确认为 **Rust-first** 项目、可克隆，已完成源码调研：

- `TimothyWalker6922/avcodec-rs-develop`（多媒体编解码 + 图像处理）—— 见 §9.1。
- `ChungTak/cheetah-media-server-rs-dev`（流媒体服务器/协议平台）—— 见 §9.2。

**关键发现（影响设计）**：两者各自维护**不同的媒体数据模型**——avcodec 用 `Image / Packet / BufferHandle / MemoryDomain`，cheetah 用自有 `cheetah-codec` 的 `AVFrame / TrackInfo / CodecId`，**cheetah 不复用 avcodec**。因此 dg-media 与 dg-stream 之间需要一层**帧模型桥接（frame bridge）**：把 cheetah 的 `AVFrame` 与 avcodec 的 `Image/Packet`、以及 dg-core 的 `Frame/Tensor` 三者互转（尽量共享底层 buffer 走 zero-copy，必要时 staging 拷贝）。这是 dg-media/dg-stream 的核心工程点，已纳入 §9 与 M5。

仍保留 **trait 边界隔离 + adapter crate** 的策略：dg-media/dg-stream 只依赖框架内 trait，具体实现指向这两个 crate，可替换/可后置。

---

## 4. 总体架构设计

### 4.1 分层视图

```
┌──────────────────────────────────────────────────────────────┐
│  应用层 / CLI / (未来) serving、gRPC/HTTP 服务                  │
├──────────────────────────────────────────────────────────────┤
│  编排层 (dg-graph)：Engine / Graph / Element / Connector       │
│    - 配置驱动(JSON/YAML) 构建 DAG，pipeline 并行、背压          │
│    - 内置 element：decode / preprocess / infer / postprocess   │
│      / tracker / osd / encode / distributor / converger        │
├───────────────┬───────────────────────────┬──────────────────┤
│ 多媒体层       │  推理运行时层 (dg-runtime)  │  调度层           │
│ (dg-media)     │   InferBackend trait        │ (dg-scheduler)    │
│  decode/encode │   + 各后端适配              │  设备/核心选择    │
│  图像处理/resize│                            │  负载均衡/亲和性   │
│  ← avcodec-rs  │                            │                   │
│ 流媒体          │                            │                   │
│ ← cheetah-...  │                            │                   │
├───────────────┴───────────────────────────┴──────────────────┤
│  后端适配 crates (feature-gated)                                │
│   dg-openvino-sys/  dg-tensorrt-sys/  dg-rknn-sys/  dg-sophon-sys│
├──────────────────────────────────────────────────────────────┤
│  核心抽象层 (dg-core)                                           │
│   DataType / DataFormat / Shape / Tensor / Buffer               │
│   Device / DeviceKind / Stream / MemoryPool / Allocator         │
│   Error / Result / Logging                                      │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Workspace / crate 划分（Cargo workspace）

| crate | 职责 | 关键依赖 |
|-------|------|----------|
| `dg-core` | 数据类型、张量、buffer、device、stream、error、log 抽象（no unsafe 对外） | `thiserror`, `half`, `bytemuck`, `tracing` |
| `dg-runtime` | `InferBackend` trait、`RuntimeOption`、后端注册表/工厂、模型加载 | `dg-core` |
| `dg-openvino-sys` / `dg-openvino` | OpenVINO C API `-sys` 绑定 + 安全封装后端 | `bindgen`, `dg-runtime` |
| `dg-tensorrt-sys` / `dg-tensorrt` | TensorRT + CUDA runtime 绑定 + 封装 | `bindgen`, `dg-runtime` |
| `dg-rknn-sys` / `dg-rknn` | rknpu2 `librknnrt` 绑定 + 封装（core_mask/量化/zero-copy） | `bindgen`, `dg-runtime` |
| `dg-sophon-sys` / `dg-sophon` | BMRuntime/bmlib/bmcv 绑定 + 封装 | `bindgen`, `dg-runtime` |
| `dg-scheduler` | 设备/核心枚举、负载均衡策略、亲和性、`auto` 分配 | `dg-core`, `dg-runtime` |
| `dg-graph` | Engine/Graph/Element/Connector/DataPipe，配置解析，pipeline 执行 | `dg-runtime`, `dg-scheduler`, `serde` |
| `dg-media` | 编解码/图像处理 adapter（对接 avcodec-rs）+ 内置 media element + frame bridge | `avcodec-rs-develop` |
| `dg-stream` | 流媒体拉流/推流 adapter（对接 cheetah）+ frame bridge | `cheetah-media-server-rs-dev` |
| `dg-elements` | 算法/工具 element 集合（yolo、tracker、osd、resnet、ppocr…） | `dg-graph`, `dg-media` |
| `dg-capi` | 稳定 **C ABI**：`extern "C"` 导出 + `cbindgen` 生成头文件；引擎/图/张量/后端的生命周期与调用接口，供多语言绑定 | `dg-graph`, `cbindgen` |
| `dg-cli` | 命令行入口：`dg run --config graph.yaml` | 全部 |

**`-sys` 与安全封装分离**（参考 nndeploy 的 backend 划分 + Rust 生态惯例）：`*-sys` 只做 FFI 绑定与链接，`build.rs` 通过 `pkg-config`/环境变量定位 SDK；安全 crate 提供 RAII、错误转换、`Send/Sync` 语义、`Drop` 释放资源。

### 4.3 编译期/运行期特性开关

- 每个后端一个 Cargo feature：`openvino` / `tensorrt` / `rknn` / `sophon`；默认全关，按目标平台开启（交叉编译到 RK3588 只开 `rknn`）。
- 部署模式 `soc` / `host` 通过 feature + 运行期 `RuntimeOption.device_mode` 双重控制（Sophon/RKNN 的 SoC vs PCIe 链接库不同）。
- 后端在编译进来时通过 `inventory`/`ctor` 静态注册到全局工厂（对应 nndeploy 的 `TypeInferenceRegister`）。

---

## 5. 核心抽象层设计（dg-core）

### 5.1 数据类型系统
借鉴 nndeploy 的 `DataType(code, bits, lanes)` 组合式设计（比固定枚举更能表达 fp4/int4/packed 等）：

```rust
pub enum TypeCode { Uint, Int, Float, Bfloat, Float8, Float4, OpaqueHandle }
pub struct DataType { pub code: TypeCode, pub bits: u8, pub lanes: u8 }
// 便捷常量：F32, F16, BF16, F8, F4, I16, U16, I8, U8, I4 ...
```

- **亚字节类型（int4/fp4）**：`bits < 8` 时按 packed 存储，`Buffer` 记录逻辑元素数与物理字节数；提供 pack/unpack 辅助。
- **量化元信息**：张量可携带 `Quantization { scheme: {None, AffineAsymmetric, Symmetric, DynamicFixedPoint}, scale: Vec<f32>, zero_point: Vec<i32>, axis }`，直接映射 RKNN 的 `qnt_type/zp/scale` 与 TensorRT/OpenVINO 的量化语义。

### 5.2 布局与形状
- `DataFormat`：`N, NC, NCHW, NHWC, NC4HW, NC8HW, NCDHW, OIHW, Auto`（对齐 nndeploy）。
- `Shape`（dims）+ `Strides`（支持 RKNN 的 `w_stride/size_with_stride` 带 stride 布局与对齐）。

### 5.3 内存三层抽象（对齐 nndeploy Device/Buffer/Tensor）
- `Device` trait：`alloc/free`、`memcpy(H2D/D2H/D2D)`、`create_stream`、`synchronize`；实现体 `CpuDevice / CudaDevice / RknnDevice / SophonDevice / OpenvinoDevice`。
- `Buffer`：带所有权与引用计数（`Arc` + RAII `Drop`）的内存块，记录 `device / memory_type(Host/Device/Unified) / desc`；支持 `clone / copy_to / from_external_ptr`（zero-copy 外部指针）。
- `Tensor`：以 `Buffer` 为载体 + `TensorDesc(name/shape/strides/dtype/format/quant/device)`；`allocate/reshape/clone/copy_to`。
- `Stream` / `Event`：异步执行与同步原语，供多核/多卡并行使用。

### 5.4 错误处理
- `thiserror` 定义 `dg_core::Error` 分层枚举（`Backend`, `Device`, `Config`, `Media`, `Io`…）；各 `-sys` 的 C 返回码转换为具体 variant，保留原始 code 与上下文；对外 `Result<T> = Result<T, Error>`。禁止 `unwrap` 于库代码路径。

---

## 6. 推理运行时与后端抽象（dg-runtime）

### 6.1 统一后端 trait
综合 nndeploy `Inference` 与 FastDeploy `BaseBackend`：

```rust
pub trait InferBackend: Send {
    fn kind(&self) -> BackendKind;
    fn init(&mut self, model: &ModelSource, opt: &RuntimeOption) -> Result<()>;
    fn reshape(&mut self, shapes: &[TensorShape]) -> Result<()>;   // 动态 shape
    fn num_inputs(&self) -> usize;
    fn num_outputs(&self) -> usize;
    fn input_info(&self, i: usize) -> TensorInfo;   // name/shape/dtype/format/quant
    fn output_info(&self, i: usize) -> TensorInfo;
    fn run(&mut self, inputs: &[Tensor], stream: Option<&Stream>) -> Result<Vec<Tensor>>;
}
```

### 6.2 RuntimeOption（统一配置入口，对齐 FastDeploy）
`backend`、`device_kind`、`device_id`、`device_mode(SoC/Host)`、`core_mask/core_selection`、`precision(fp32/fp16/int8…)`、`cpu_thread_num`、`external_stream`、`model_format`、`enable_zero_copy`、`dynamic_shape` 等。

### 6.3 各后端适配要点
- **OpenVINO**：`ov::Core → read_model → compile_model(device: "CPU"/"GPU"/"NPU"/"AUTO"/"MULTI") → InferRequest`；精度经 `ov::hint::inference_precision` 或模型自带量化；多设备靠 `AUTO/MULTI` 插件与 device_id。
- **TensorRT**：加载 `.engine`（或 `onnx`→builder，首期只支持预构建 engine）；`IExecutionContext` + CUDA stream；fp16/int8 由 engine 决定；多 GPU 靠 `cudaSetDevice(device_id)` + 每设备独立 context。
- **RKNN2**：`rknn_init → query(IN_OUT_NUM/INPUT_ATTR/OUTPUT_ATTR)`；`rknn_set_core_mask` 做核心选择/负载均衡；量化用 `rknn_tensor_attr` 的 `qnt_type/zp/scale`；`rknn_create_mem`+`rknn_set_io_mem` 走 zero-copy；`rknn_set_input_shapes` 动态 shape。SoC 模式链接板端 `librknnrt`。
- **Sophon**：`bmrt_create → bmrt_load_bmodel → bmrt_launch`；`bmlib` 管理 device memory（`bm_malloc_device_byte_heap`/`bm_free_device`/`bm_memcpy_*`）；`bmcv` 做前处理与 resize；PCIe(Host) 与 SoC 两套链接；多芯片靠 `bm_handle` per device。

### 6.4 数据类型 × 后端支持矩阵（首期目标）

| 精度 | OpenVINO | TensorRT | RKNN2 | Sophon |
|------|:--:|:--:|:--:|:--:|
| fp32 | ✅ | ✅ | ✅ | ✅ |
| fp16 | ✅ | ✅ | ✅ | ✅ |
| bf16 | ✅ | ✅(新) | ⚠️ | ⚠️ |
| int8 | ✅ | ✅ | ✅ | ✅ |
| uint8 | ✅ | ⚠️(IO) | ✅ | ✅ |
| int16 | ⚠️ | ⚠️ | ✅ | ⚠️ |
| uint16 | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| int4 | ⚠️(新) | ✅(新 GPU) | ⚠️ | ⚠️ |
| fp8 | ⚠️ | ✅(Hopper+) | ❌ | ⚠️ |
| fp4 | ❌ | ⚠️(Blackwell) | ❌ | ❌ |

> ✅ 明确支持；⚠️ 视 SDK 版本/芯片型号而定，需运行期能力探测；❌ 暂不支持。框架**统一表达所有类型**，实际可用性由“后端能力查询（capability query）”在 `init` 时校验并给出清晰报错，而非静默失败。

### 6.5 端到端零拷贝（硬件对象直连推理，重点）
目标：**从 dg-media（avcodec-rs）的硬件解码/图像处理输出，直接喂入推理后端，全程不经过 CPU 拷贝**。

设计要点：
- **统一 buffer 句柄贯穿全链路**：dg-core `Buffer` 记录 `MemoryType` 与底层句柄（`DmaBuf fd` / `DrmPrime` / `VaapiSurface` / `CudaDevicePtr` / `MppBuffer` / Sophon device addr）。avcodec 的 `MemoryDomain` / `BufferHandle` 与之一一映射，导入时**共享而非复制**（`Buffer::from_external`，用 `ExternalDropGuard` 托管生命周期与引用计数）。
- **后端零拷贝入口**：
  - RKNN：`rknn_create_mem_from_fd` / `rknn_set_io_mem` 绑定 dma-buf，配合 native NHWC layout 与 `w_stride/size_with_stride`；
  - Sophon：解码/`bmcv` 输出的 device memory 直接作为 `bmrt_launch` 的输入张量（同一 `bm_handle`）；
  - TensorRT：NVDEC/`cuvid` 输出的 CUDA device ptr 直接作为绑定输入（同一 CUDA context/stream）；
  - OpenVINO：`remote tensor`（VA-API/GPU surface）或 host zero-copy `ov::Tensor(ptr)`。
- **前处理融合**：resize/csc/normalize 尽量用硬件（RGA/bmcv/VPP/libyuv）在设备侧完成，输出直接是后端期望的 layout/dtype，避免“解码→CPU→前处理→CPU→推理”的多次搬运。
- **兜底 staging**：当源内存域与目标后端不兼容（跨卡、跨异构设备）时，才走 avcodec 的 `StageHook` 显式 staging；框架据能力探测**自动选择 zero-copy 或 staging 路径**，并在日志中标注实际路径与拷贝次数，便于性能诊断。
- **约束**：zero-copy 要求 dg-media 解码器与推理后端**运行在同一物理设备/上下文**；调度器（§7）在做设备/核心分配时把“与解码同设备”作为亲和性偏好之一。

---

## 7. 设备调度与负载均衡（dg-scheduler）

### 7.1 设备/核心模型
```rust
pub enum DeviceKind { Cpu, IntelGpu, IntelNpu, CudaGpu, RknnNpu, SophonTpu }
pub struct DeviceId { pub kind: DeviceKind, pub card: u16, pub core: CoreSel }
pub enum CoreSel { Auto, Single(u8), Mask(u32), All }   // 映射 RKNN core_mask / Sophon 多核
```

### 7.2 负载均衡策略
- `auto`：调度器根据实时负载（每设备/核心的在途任务数、队列深度）选择最空闲的设备/核心 —— round-robin / least-loaded / 亲和性（同一路流尽量固定核心以利用缓存与上下文复用）。
- **显式指定**：配置里为某 element 指定 `device_id` + `core`（例如把两个模型实例分别绑定 RK3588 的 core0、core1）。
- **单卡多核**：为同一模型创建多个后端实例（每实例绑定不同 core_mask），调度器在实例间分发帧。
- **多卡多核**：设备发现枚举所有卡与核心，形成资源池，统一按策略分发。
- **RKNN 特例**：优先用 `RKNN_NPU_CORE_AUTO` 让 runtime 自动均衡；需要确定性时用显式 core_mask + 多 context。

### 7.3 SoC vs Host
- 通过 `RuntimeOption.device_mode` 与 feature 决定链接库与内存路径（SoC 下常有统一内存/物理连续内存可 zero-copy；Host/PCIe 需显式 H2D/D2H）。设备抽象层屏蔽差异，上层无感。

---

## 8. 图执行引擎（dg-graph）

### 8.1 模型（融合 sophon-stream + nndeploy）
- `Engine`：进程级，管理多个 `Graph`。
- `Graph`：一张 DAG（可嵌套，`Graph` 亦是可执行节点）。
- `Element`（≈ Node）：最小处理单元，多线程 + 多输入 DataPipe。
- `Connector` + `DataPipe`：有容量上限的类型化队列（默认容量可配，参考 sophon-stream 默认 20），满则背压（软阻塞 + 上报事件）。
- `Group`：把算法的 pre/infer/post 合并为一个逻辑 element（简化配置，源自 sophon-stream）。
- 并行模式 `ParallelType`：`Sequential / Task / Pipeline`（源自 nndeploy）。
- `ThreadPool`：work-stealing（`tryPush/tryPop/trySteal`），线程数与输入 DataPipe 数对应。

### 8.2 数据载荷
- 帧对象 `Frame` / `ObjectMetadata`：携带 `channel_id`、图像 buffer（zero-copy 引用 device buffer）、张量、检测/跟踪结果等，沿链路传递（对齐 sophon-stream 的 `ObjectMetadata`）。

### 8.3 配置模型（自研 schema，非直接对齐 engine.json）
sophon-stream 的 `engine.json` 有明显不足：节点用魔法数字 `id`（5000/5001…）、`connections` 用 `src_id/src_port` 手工连线易错、无 schema 版本、无法表达可复用模板与参数化、不利于动态生成与热更新。本项目**设计更合理的配置模型**：

**核心原则**
- **格式无关**：内部只有一份强类型 `GraphSpec`（Rust struct + `serde`）；外部支持 **YAML（默认）/ JSON / TOML**，可选 XML（通过 `quick-xml`）。序列化/反序列化对称，任一格式可互转。
- **稳定标识**：节点用**字符串 `name`（图内唯一）**而非魔法数字；端口用具名端口 `node.port`（如 `decode.video_out`）而非数字下标。
- **声明式连线**：`edges: ["decode.out -> yolo.in", ...]`（人类可读），解析为内部 typed edge；杜绝手工对齐 src/dst 数字。
- **可复用与参数化**：支持 `templates`（可复用子图/element 模板）+ `${var}` 变量与 `defaults` 合并 + `include` 引用外部片段；便于多路流复用同一算法配置（吸收 sophon-stream `Group` 的意图但更通用）。
- **版本化**：顶层 `apiVersion` + `kind`（借鉴 k8s 风格），保证向后兼容与迁移。
- **可校验**：提供 JSON Schema 导出 + 加载期校验（未知字段、类型、端口连通性、DAG 无环、后端/精度可用性 preflight），错误定位到具体节点/字段。

**动态生成 / 修改 / 热加载（一等能力）**
- `GraphSpec` 是纯数据，可由代码/UI/上游系统**程序化构建**并序列化输出（动态生成）。
- **Builder API**：`GraphBuilder::new().add_node(...).connect(...).build()`；C ABI 侧同样暴露增删节点/连边接口。
- **热更新**：Engine 支持 `apply(patch)` / `reload(spec)` —— 对运行中的 Graph 做**增量 diff**（新增/删除/替换节点与边、改参数），无需整体重启；对无法热改的节点走安全的局部 drain + rebuild。配置文件变更可选 `watch` 自动触发。

**示例（YAML）**
```yaml
apiVersion: dg/v1
kind: Graph
defaults: { device: { kind: OpenVino, id: 0 }, precision: fp16 }
vars: { model_dir: /opt/models }
nodes:
  - name: cam0
    type: rtsp_source
    params: { url: "rtsp://..." }
  - name: decode
    type: decode
    threads: 4
    params: { hw: auto, zero_copy: true }
  - name: yolo
    type: yolov8
    backend: openvino          # 覆盖 defaults；M4 起可切 rknn
    precision: int8
    params: { model: "${model_dir}/yolov8.xml" }
  - name: track
    type: bytetrack
  - name: osd
    type: osd
  - name: enc
    type: encode
    sink: true
edges:
  - cam0.out    -> decode.in
  - decode.image-> yolo.image
  - yolo.dets   -> track.dets
  - track.out   -> osd.in
  - osd.out     -> enc.in
```
源节点（source/decode）由应用层通过 `engine.push_input(channel_task)` 或直接由 stream source 驱动。

### 8.4 内置 element（分批实现）
- multimedia：`decode` / `encode` / `osd` / `resize`（走 dg-media）。
- algorithm：`yolov5/8` / `resnet` / `bytetrack` / `ppocr` / `retinaface` …（走 dg-elements + dg-runtime）。
- tools：`distributor`（分发）/ `converger`（汇聚）/ `filter` / `http_push`。
- stream：`rtsp_pull` / `rtmp_push` / `webrtc`（走 dg-stream）。

---

## 9. 多媒体与流媒体集成

### 9.1 多媒体（dg-media，依赖 avcodec-rs-develop）
avcodec-rs 是 Rust-first 媒体中台：core 层 Sans-I/O，`Decoder/Encoder/ImageProcessor/CaptureSource` trait + `Poll{Ready/Pending/EndOfStream}` 非阻塞模型 + `RegistryBuilder/Registry` 的**可解释后端选择**（capability 匹配 + preflight + trace），并有 `Image/Packet/BufferHandle/MemoryDomain(Host/DmaBuf/DrmPrime/Vaapi/Cuda/Mpp…)` 的 zero-copy + staging hook 抽象。

- **对接策略**：dg-media 直接走它的 **Rust API**（不走 `avrs_*` C ABI）——用 `RegistryBuilder` 注册所需 backend（`backend_ffmpeg` / `backend_rkmpp` / `backend_nvcodec` / `backend_libyuv` / `backend_video_device`），把它的 `Decoder/Encoder/ImageProcessor` 适配到 dg-media 的内置 element。
- **精度/内存对齐**：avcodec 的 `MemoryDomain` 与 dg-core 的 `Device/Buffer` 对齐（DmaBuf/Mpp→RknnDevice、Cuda→CudaDevice…），解码输出尽量 zero-copy 直连推理前处理；跨域用它的 `StageHook` staging。
- **注意**：它的 FFmpeg 后端是 **codec-only**（libavcodec + swresample，非 libavformat 中心），解复用/封装能力有限——容器解复用主要由 dg-stream（cheetah）承担。
- **错误归一化**：把 `AvError{Kind/Detail}` 映射到 dg-core `Error::Media`。

### 9.2 流媒体（dg-stream，依赖 cheetah-media-server-rs-dev）
cheetah 是完整的 Rust 流媒体平台：协议**三段式 core / driver-tokio / module**，覆盖 `rtmp/rtsp/http-flv/hls/ts/rtp/srt/gb28181/fmp4/mp4/webrtc`；媒体基础是自有 `cheetah-codec`（`AVFrame/TrackInfo/CodecId` + 各类 demux/mux）；既可嵌入（`cheetah-sdk`）也可独立服务（`apps/cheetah-server`）。

- **对接策略**：dg-stream 走 `cheetah-sdk` 的**嵌入式 API**：
  - 拉流 → 适配 `SubscriberSource`（`recv()->Arc<AVFrame>` / `close()`），经 `SubscriberApi::subscribe` / `StreamManagerApi::open_subscriber`；
  - 推流 → 适配 `PublisherSink`（`update_tracks` / `push_frame(Arc<AVFrame>)` / `close`），经 `PublisherApi::acquire_publisher` / `StreamManagerApi::open_publisher`；
  - 更底层协议桥接用 `CoreAdaptersApi`（`publish_frame/update_tracks/close_stream`）。
- **frame bridge（关键）**：cheetah 的 `AVFrame` ↔ avcodec 的 `Image/Packet` ↔ dg-core `Frame/Tensor` 三者互转，尽量共享底层 buffer 走 zero-copy，跨内存域必要时 staging。这是 dg-media/dg-stream 的核心工程点。
- **部署形态**：首期以嵌入式库集成（拉流/推流 element）；后续可选把整个 dg-graph 引擎注册为 cheetah 的一个 `Module`，复用其 control/engine/module-manager 做服务化。

---

## 10. FFI、构建与交叉编译

### 10.1 FFI 绑定
- `bindgen` 从各 SDK 头文件生成 `-sys`（对齐 avcodec-rs 做法：优先提交生成好的 bindings，减少对 libclang 的构建期依赖）；`build.rs` 用 `pkg-config` 或 `DG_<BACKEND>_SDK_DIR` 环境变量定位库与头文件，`cargo:rustc-link-lib` 链接。
- **供应链**：新增第三方依赖优先选发布 ≥7 天、无 floating range 的版本。

### 10.2 交叉编译（x86_64 / aarch64 / Android / RISC-V）
- 目标三元组矩阵：`x86_64-unknown-linux-gnu`、`aarch64-unknown-linux-gnu`（RK3588、算能 SE）、`aarch64-linux-android`（NDK）、`riscv64gc-unknown-linux-gnu`，及 Host/SoC 变体。
- 用 `cross` / `cargo-zigbuild`（zig 作为跨平台 C 交叉工具链，简化 sysroot）+ 每目标固化 sysroot 与各厂商板端 SDK（`librknnrt`/BMRuntime 的对应 arch 版本）。
- 各后端 feature 与目标平台绑定：交叉编译到 RK3588 只开 `rknn`；x86 host 开 `openvino`（可选 `tensorrt`/`sophon-pcie`）。CI 构建矩阵覆盖全部三元组（见 §12）。
- **能力探测**：运行期查询 SDK 版本与设备能力，写入日志，用于精度/核心可用性校验。

### 10.3 稳定 C ABI（dg-capi，多语言绑定基座）
- `dg-capi` crate 用 `extern "C"` 暴露稳定 ABI：引擎/图/节点/张量/后端的**创建、配置、增删连边、push 输入、poll 输出、销毁**，以及 zero-copy 的**外部 buffer 导入**（传入 dma-buf fd / device ptr）。
- 用 `cbindgen` 生成 C 头文件作为一等交付物；错误以整型 code + 线程局部 `dg_last_error()` 字符串返回（对齐 avcodec `avrs_*` 风格）。句柄用不透明指针 + 显式 `*_free`，跨 ABI 不暴露 Rust 类型。
- 首期只交付 C API 与头文件（不做 Python/serving），为后续 Python(ctypes/PyO3)、Go(cgo)、C++ 绑定预留。

### 10.4 可观测性
- `tracing` 结构化日志 + 每 element 的吞吐/时延/队列深度指标（后续可接 Prometheus）。

### 10.5 Rust 工程最佳实践（借鉴 avcodec-rs）
明确采纳 avcodec-rs 中已验证的实践：
- **Sans-I/O 核心**：dg-core / dg-runtime / dg-graph 的核心逻辑（协议状态机、图调度、张量运算）**不直接做 I/O 与阻塞**，I/O 由外层 driver/adapter 注入（对齐 avcodec 的 core + cheetah 的 core/driver 分层）。核心因此易测、可确定性重放、可跨 async 运行时移植。
- **非阻塞 Poll 模型**：推理/编解码/流的提交-轮询采用 `Poll{Ready/Pending/EndOfStream}` 风格，避免线程阻塞，适配 pipeline 并行与背压。
- **分层错误模型**：`Error{kind, detail}` + 可映射整型码（对齐 avcodec `AvError`），保留上下文，禁止库路径 `unwrap`。
- **可解释的能力/后端选择**：backend selection 走“能力匹配 + preflight + trace + cache”，失败给出可读诊断，而非静默回退（对齐 avcodec `Registry`）。
- **测试策略**：
  - 单元 + 集成测试（mock 后端，无硬件可跑）；
  - **属性测试**（`proptest`）：张量 pack/unpack、DataType/DataFormat 转换、配置 round-trip（YAML/JSON/TOML 互转不丢信息）、图 diff/热更新不变量；
  - **fuzz 测试**（`cargo-fuzz`/libFuzzer）：配置解析器、C ABI 边界入参、模型/码流解析等不可信输入面；
  - 精度回归：固定输入比对各后端输出（余弦相似度阈值）。
- **安全边界**：`unsafe` 仅限 `-sys`/FFI adapter，集中审查并注释不变量；对外 crate `#![forbid(unsafe_code)]`（dg-core/dg-graph 等）。`clippy`（含 `pedantic` 选择性开启）+ `rustfmt` + `cargo-deny`（许可/漏洞/供应链）进 CI。

---

## 11. 分阶段开发计划（里程碑）

> 每个里程碑均要求：编译通过、`cargo clippy`/`fmt` 零告警、单元测试、文档更新。硬件相关部分在无硬件时用 mock 后端 + 录制数据做 CI，在目标板/卡上做集成验证。

### M0 — 项目骨架与核心抽象（1–1.5 周）
- 建立 Cargo workspace 与上述 crate 骨架、CI（fmt/clippy/test/构建矩阵）、feature 开关。
- `dg-core`：DataType/DataFormat/Shape/Tensor/Buffer/Device(Cpu)/Stream/Error/log。
- 交付：`cargo build` 全绿；CpuDevice 上张量分配/拷贝/reshape 单测。

### M1 — 运行时抽象 + 首个后端 OpenVINO（2 周）
- `dg-runtime`：`InferBackend` trait、`RuntimeOption`、后端注册工厂、`ModelSource`；mock 后端供 CI。
- **首个后端 OpenVINO**（x86 CPU 即可运行，CI 最友好）打通端到端。
- 交付：加载模型 → 单张推理 → 输出正确；后端能力查询 + 精度校验报错清晰。

### M2 — 图执行引擎 + 配置模型（2–2.5 周）
- `dg-graph`：Engine/Graph/Element/Connector/DataPipe、ThreadPool(work-stealing)、ParallelType、背压。
- 配置模型（§8.3）：`GraphSpec` + YAML/JSON/TOML 加载、校验、Builder API、round-trip 测试（热更新放 M6 增强）。
- 内置 `input`/`infer`/`sink` 最小 element，跑通单图。
- 交付：最小 pipeline 端到端 + 背压/多线程正确性 + 配置 round-trip 属性测试。

### M3 — 第二后端 RKNN2 + 调度与负载均衡（2.5–3 周）
- **第二后端 RKNN2**（用户指定次序：先 OpenVINO 再 RKNN2）：core_mask、量化元信息、zero-copy(`rknn_create_mem`/`set_io_mem`)、动态 shape；aarch64 SoC 交叉编译。
- `dg-scheduler`：设备发现、DeviceId/CoreSel、`auto`（least-loaded/round-robin/亲和性，含“与解码同设备”偏好）与显式指定；单卡多核多实例分发。
- 交付：RK3588 三核负载均衡实测（利用率/吞吐数据）；RKNN sample 通过。

### M4 — 补齐后端矩阵 + SoC/Host 双模式（2.5–3 周，可并行）
- 补齐 **TensorRT / Sophon**，SoC/Host 双模式，数据类型矩阵按 §6.4 落地 + 能力探测。
- 交付：四后端各自 sample 通过；数据类型/精度矩阵验证报告。

### M5 — 多媒体与流媒体 + 端到端零拷贝（3–4 周）
- `dg-media`（avcodec-rs adapter）：decode/encode/resize/osd element，硬件编解码路径。
- `dg-stream`（cheetah adapter）：rtsp/http-flv 拉流、rtmp/webrtc 推流 element。
- **frame bridge**（§9）+ **端到端零拷贝**（§6.5）：硬件解码对象直连推理，路径诊断日志。
- 交付：“RTSP 拉流→硬解→前处理→YOLO 推理→跟踪→OSD→RTMP 推流”零拷贝 demo + 拷贝次数对比。

### M6 — C ABI、算法库、CLI、热更新与打磨（3–4 周）
- `dg-capi`（§10.3）：稳定 C ABI + `cbindgen` 头文件 + C 示例（多语言绑定基座）。
- 配置**热更新**（`apply/reload`、增量 diff、`watch`）。
- `dg-elements`：yolov5/8、resnet、bytetrack、ppocr、retinaface、distributor/converger 等。
- `dg-cli`：`dg run --config graph.yaml`；fuzz 目标（配置/C ABI/码流解析）、benchmark、跨平台发布产物。
- 交付：多路流多算法综合 demo + 用户指南 + C API 示例 + benchmark。

**总周期**：约 **15–20 周**（后端矩阵与算法库可并行压缩）。跨平台交叉编译（x86_64/aarch64/Android/RISC-V）贯穿各里程碑，随对应后端落地在 CI 构建矩阵中逐步开启。

---

## 12. CI / 测试策略
- **无硬件层**：mock 后端 + 单元/集成测试跑在通用 x86 runner；`fmt`/`clippy`/`cargo test`/各 target `cargo check`。
- **有硬件层**：OpenVINO（x86 CPU，CI 可跑）、RKNN（RK3588 自托管 runner）、TensorRT（NVIDIA runner）、Sophon（BM 卡/SE 盒 runner）—— 需用户提供或授权自托管 runner。
- 精度回归：固定输入 → 比对各后端输出与参考（余弦相似度阈值）。

## 13. 主要风险与缓解
| 风险 | 缓解 |
|------|------|
| avcodec 与 cheetah 媒体模型不一致（Image/Packet vs AVFrame） | 引入 frame bridge 统一转换，zero-copy 优先 + staging 兜底（见 §9） |
| 各 SDK 版本/精度支持差异大（fp4/int4/uint16） | 运行期能力探测 + 清晰报错，统一类型表达但不强行支持 |
| 无目标硬件难以验证 | mock 后端 + 自托管 runner，用户提供板/卡 |
| 亚字节类型（int4/fp4）packed 复杂 | dg-core 统一 pack/unpack + 充分单测 |
| 交叉编译工具链复杂（x86/aarch64/Android/RISC-V） | `cross`/`cargo-zigbuild` + 固化 sysroot + 构建矩阵文档化（§10.2） |
| 端到端零拷贝跨异构设备不总成立 | 能力探测自动选 zero-copy/staging，日志标注实际路径与拷贝次数（§6.5） |

## 14. 已确认决策
1. **依赖仓库**：avcodec-rs-develop / cheetah-media-server-rs-dev 已确认可访问并完成调研（见 §9）。
2. **后端次序**：先 **OpenVINO**（M1）再 **RKNN2**（M3），后续 TensorRT/Sophon（M4）。
3. **部署目标**：Host 与 SoC **两者都要**；交叉编译覆盖 **x86_64 / aarch64 / Android / RISC-V**（见 §1.1-8、§10.2）。
4. **多语言/服务**：首期**不做** Python 绑定与 serving，但交付稳定 **C ABI + 头文件**（`dg-capi`，§10.3）为后续绑定预留。
5. **文档**：本设计文档保存为仓库初始 `docs/design.md`（本 PR）。

### 后续可再确认（不阻塞启动）
- CI 自托管 runner（RKNN/TensorRT/Sophon 硬件）由谁提供；配置格式默认 YAML 是否认可；算法 element 首批清单优先级。
