# `docs/design.md` 剩余任务清单

> 审计基线：`main` @ `1498dd2`（2026-07-10）
>
> 执行规则：下表每个任务使用一个独立 PR；实现 PR 必须同步更新本文件状态，并通过
> `fmt`、`clippy`、workspace tests 以及受影响的交叉编译检查。

## 状态说明

- `未开始`：尚无满足验收条件的实现。
- `进行中`：已有对应 PR，但尚未合并或未满足全部验收条件。
- `外部阻塞`：软件准备工作可继续，但最终验收需要 SDK、设备或自托管 runner。
- `可选`：`docs/design.md` 明确标注为可选，不阻塞首期完成。
- `已完成`：代码和验证均已合并；不再属于剩余任务。

## 已完成基线

以下能力已经落地，不重复列为剩余任务：

- `dg-core` 基础数据类型、量化、shape/stride、CPU tensor/buffer，以及 int4/fp4
  pack/unpack 属性测试。
- mock/OpenVINO/RKNN/TensorRT/Sophon 后端骨架、静态注册、通用 `inference`
  element 和静态 preflight。
- GraphSpec YAML/JSON/TOML、include/template/variable、Builder、DAG/端口名称校验、
  diff/reload/watch 基础 API 和 round-trip/property tests。
- Sequential/Task/Pipeline 执行、work-stealing pool、有界队列背压。
- C ABI 基础生命周期、图编辑、push/run/poll、diff/reload、外部句柄入口、cbindgen
  头文件和 C 示例。
- 现有算法 element、CLI 基础命令、配置/C ABI fuzz target、primitive benchmark、
  四目标默认 feature CI 和发布归档。
- `source`、`input`、`mock_inference`、`sink`、通用 `inference` 的加载期参数校验。

## A. 配置模型与图执行

| ID | 状态 | 独立 PR 范围 | 验收条件 | 依赖 |
|---|---|---|---|---|
| CFG-01 | 已完成 | GraphSpec 严格端口连通与基数校验 | 加载期拒绝缺失必需输入、同一输入多条入边和重复边；错误定位到具体 connection/port，而不是运行时失败 | 无 |
| CFG-02 | 已完成 | `dg-media` 四个 element 参数 validator | `media_decode/encode/resize/osd` 拒绝未知字段、错误类型、零尺寸和非法 box/color；创建与校验复用同一解析逻辑 | 无 |
| CFG-03 | 已完成 | `dg-stream` 四个 element 参数 validator | `rtsp_src/httpflv_src/rtmp_sink/webrtc_sink` 在加载期校验 URL、协议、队列、背压、track 和未知字段，不打开网络连接 | 无 |
| CFG-04 | 已完成 | `dg-elements` 十个算法/并行 element 参数 validator | yolo、resnet、retinaface、bytetrack、ppocr、distributor、converger 全部使用 validator；非法阈值/尺寸/策略和未知字段在加载期报错 | 无 |
| CFG-05 | 已完成 | element 专属参数 JSON Schema 与用户导出入口 | registry 提供参数 schema；导出 schema 按 element kind 描述 required/type/enum；新增 `dg schema` 并测试所有注册项均有 schema | CFG-02、CFG-03、CFG-04 |
| CFG-06 | 已完成 | GraphSpec `defaults` 合并 | 支持全局 device/precision/backend 默认值；node override 优先；与 template/variable/include 的优先级有测试和文档 | 无 |
| CFG-07 | 已完成 | 对齐 §8.3 标准字段 | 支持或兼容 `type`/`edges`，并使设计文档中的规范示例可解析；保留现有 `kind`/`connections` 的兼容性 | CFG-06 |
| CFG-08 | 已完成 | per-node `threads` 与 `sink` 语义 | `threads >= 1` 控制 element 实例/worker；`sink` 明确终止输出连通要求；配置、执行和错误路径有测试 | CFG-01、CFG-07 |
| CFG-09 | 已完成 | template/variable/include 引用严格校验 | 未知 template、未解析 `${var}`、无 base dir 的 include 和 include 环均在加载期给出字段级错误 | 无 |
| HOT-01 | 已完成 | CLI 文件 watch 入口 | `dg run --watch` 监听配置、输出 diff、拒绝非法 reload，并更新用户指南 | 无 |
| HOT-02 | 未开始 | 运行中 Graph 的增量热更新 | 对运行中的图增删/替换节点与边；未受影响节点保持运行；不可热改节点安全局部 drain + rebuild；有状态连续性测试 | CFG-01、HOT-01 |
| CFG-10 | 可选 | XML 配置支持 | 仅在确认需要时通过 `quick-xml` 增加 XML 加载和 round-trip property test | 无 |

## B. 核心、运行时、后端与调度

| ID | 状态 | 独立 PR 范围 | 验收条件 | 依赖 |
|---|---|---|---|---|
| CORE-01 | 进行中 | 完成 device/memory/stream 抽象 | 增加可注册的非 CPU Device/Stream/Event adapter 与 MemoryPool/Allocator；上层不直接依赖厂商 FFI | SYS-01 至 SYS-04 |
| SYS-01 | 进行中 | `dg-openvino-sys` 分层 | FFI/link 只在 `-sys`；`dg-openvino` 保持 safe wrapper；默认构建无 SDK | 无 |
| SYS-02 | 进行中 | `dg-rknn-sys` 分层 | bindgen/build/link/unsafe 移入 `dg-rknn-sys`；安全 crate 仅 RAII 与 `InferBackend` | 无 |
| SYS-03 | 进行中 | `dg-tensorrt-sys` 分层 | TensorRT/CUDA shim、bindings、link 和 raw calls 移入 `dg-tensorrt-sys` | 无 |
| SYS-04 | 进行中 | `dg-sophon-sys` 分层 | BMRuntime/bmlib bindings、link 和 raw calls 移入 `dg-sophon-sys` | 无 |
| BE-01 | 未开始 | RKNN/Sophon 无硬件 adapter type-check | stub sys 覆盖真实 backend 模块，而非只测纯转换函数；默认 CI 能发现 adapter 编译回归 | SYS-02、SYS-04 |
| RT-01 | 进行中 | 补齐统一 `RuntimeOption` 与 stream-aware inference API | 支持 device_id/core selection、cpu threads、model format、external stream、zero-copy/dynamic-shape 通用入口；`InferBackend` 提供非阻塞 submit/poll 或等价 stream API | CORE-01 |
| RT-02 | 未开始 | 运行期 SDK/设备能力探测 | 各后端 init 查询 SDK 版本、设备、精度和部署能力；静态表仅为无硬件 fallback；不支持时给出明确诊断且不静默降级 | SYS-01 至 SYS-04 |
| SCH-01 | 未开始 | 设备发现与 scheduler/runtime 接线 | 枚举设备/核心形成 topology；Graph inference 创建后端前获取 lease，并把 device/core/deploy mode 写入 RuntimeOption | RT-01、RT-02 |
| SCH-02 | 未开始 | 多实例负载均衡 | 同模型按 core/card 创建实例池；支持 least-loaded、round-robin、显式绑定和 stream affinity；lease 生命周期反映在途负载 | SCH-01 |
| MEM-01 | 未开始 | 真正的外部设备 buffer | `Buffer` 可只持 dma-buf/device ptr 而不分配等长 host Vec；host 访问必须显式 map/stage；C ABI 导入保持 RAII 所有权 | CORE-01、SYS-02 至 SYS-04 |
| MEM-02 | 未开始 | 各后端 external-buffer zero-copy 入口 | RKNN `create_mem_from_fd`、TensorRT CUDA ptr、Sophon device mem、OpenVINO remote/host tensor 按能力直接绑定；不兼容时 staging 并记录 copy count | MEM-01、RT-02 |
| CAPI-01 | 未开始 | C ABI 后端直接生命周期与能力接口 | 提供 backend 创建/配置/能力查询/销毁，不要求调用者必须构造 Graph；同步头文件、错误码与 C 示例 | RT-01、RT-02 |

> SYS-01 说明：社区 `openvino` crate 的 FFI/link 依赖已隔离到
> `dg-openvino-sys`；`dg-openvino` 仅保留 `#![forbid(unsafe_code)]` 的安全
> wrapper。默认构建不启用 backend，因此仍不需要 OpenVINO SDK。

> SYS-02 说明：RKNN 的 bindgen、SDK 定位和 `rknnrt` 链接已隔离到
> `dg-rknn-sys`；`dg-rknn` 保留 RAII、错误转换和 `InferBackend` 安全适配。
> 默认构建不启用 backend，因此仍不需要 RKNN SDK。

> SYS-04 说明：Sophon 的 bindgen、SDK 定位和 `bmrt`/`bmlib` 链接已隔离到
> `dg-sophon-sys`；`dg-sophon` 保留 RAII、错误转换和 `InferBackend` 安全适配。
> 默认构建不启用 backend，因此仍不需要 Sophon SDK。

> SYS-03 说明：TensorRT/CUDA 的 C++ shim、bindgen 和链接已隔离到
> `dg-tensorrt-sys`；`dg-tensorrt` 保留 RAII、错误转换、`InferBackend`
> 安全适配和 SDK-free 的 `mock_sys` 测试路径。默认构建不启用 backend，
> 因此仍不需要 CUDA 或 TensorRT SDK。

> CORE-01 说明：`dg-core` 通过 inventory 注册 Device/Stream/Event adapter，
> 并提供 CPU 参考实现的 `MemoryPool`/`Allocator`；默认构建保持 SDK-free，
> 厂商设备 adapter 延后到后续 MEM-*/RT-* 任务。

> RT-01 说明：`dg-runtime` 增加统一 `RuntimeOption` 字段、`run_with_stream`
> 以及 `Runtime` 的 submit/poll 入口；厂商字段映射延后到 RT-02、SCH-01 和
> MEM-02。

## C. 多媒体、流媒体与 element

| ID | 状态 | 独立 PR 范围 | 验收条件 | 依赖 |
|---|---|---|---|---|
| APP-01 | 已完成 | CLI/C API 链接 media/stream registry | feature-gated 链接 `dg-media`/`dg-stream`；`list-elements` 和配置加载能发现八个 element；默认 build 仍无外部 SDK | CFG-02、CFG-03 |
| MEDIA-01 | 已完成 | avcodec-rs 真实 adapter | `avcodec` feature 通过 RegistryBuilder 驱动 Decoder/Encoder/ImageProcessor；x86 software codec 测试覆盖真实码流；AvError 映射完整 | APP-01 |
| STREAM-01 | 已完成 | cheetah 真实 connector | 提供可安装的 embedded `CheetahRuntimeConnector`，实现 RTSP/HTTP-FLV pull 和 RTMP/WebRTC push；本地 loopback 集成测试通过 | APP-01 |
| STREAM-02 | 进行中 | cheetah frame 元数据保真 | push/pull 保留 track id、media kind、codec、format、timebase、PTS/DTS 与 extradata，不再写死 Unknown/Data | STREAM-01 |
| MEDIA-02 | 未开始 | frame bridge 与 planner 接入真实数据路径 | avcodec Image/Packet、cheetah AVFrame、dg-core Buffer/Tensor 共享兼容句柄；staging fallback 显式记录域、路径、copy count | MEDIA-01、STREAM-02、MEM-01 |
| ELEM-01 | 未开始 | `filter` element | 注册可配置、可验证、Sans-I/O 的 filter；覆盖 pass/drop 和未知字段测试 | CFG-04 |
| ELEM-02 | 未开始 | `http_push` element | 注册可配置 HTTP sink/driver；请求失败明确报错；网络 I/O 与 element 核心逻辑分层并可注入测试 driver | CFG-04 |

> MEDIA-01 说明：已实现真实 JPEG/MJPEG 与视频 decode/encode adapter。`dg-media` 通过
> `default_registry_builder()` 注册当前编译进来的 avcodec 后端，并使用 `backend_hint` 按
> 「跟随推理硬件优先、软件回退」顺序选择：`rkmpp`、`nvcodec`、`onevpl`、`amf`，再回退到
> `ffmpeg`/`x264`/`openh264`。这些原生后端均 feature-gated 且默认关闭；默认构建只启用
> `jpeg`/`zune` 视频编解码路径（另启用 SDK-free 的 `libyuv` CSC backend），因此 SDK-free
> 交叉编译路径不依赖 FFmpeg 或硬件 SDK。`rust-h264`/
> `native-free-software` 接线已移除。
>
> 对硬件解码可能产生的 YUV 图像，颜色空间转换不在 `dg-media` 手写实现，而是委托给
> avcodec 的 `ImageProcessor` `Csc` 操作；该选择同样遵循解码硬件偏好，Rockchip/Auto
> 优先尝试 `librga`，再回退到无外部 SDK 构建依赖的 `libyuv` 软件 backend，转换为
> `Rgb24` 后进入下游 `MediaFrame`。若运行时没有可用 CSC processor，则显式返回错误，
> 不静默降级。详见
> [docs/upstream/avcodec-rs-media01-requirements.md §0](upstream/avcodec-rs-media01-requirements.md)。

> STREAM-02 说明：cheetah 每帧的 track id、媒体类型、codec、格式、timebase、
> PTS/DTS 与关键帧标志存放在 `dg-media` 的帧元数据中；push 侧优先使用帧元数据，
> 缺失时按帧的 track id 查询已公告的 TrackInfo 缓存，并在无法解析时显式报错，
> 不写入 `Unknown`/`Data` 作为静默回退。

## D. 可观测性、测试与交付

| ID | 状态 | 独立 PR 范围 | 验收条件 | 依赖 |
|---|---|---|---|---|
| OBS-01 | 未开始 | element 运行指标 | 每节点输出吞吐、处理时延、队列深度、drop/backpressure 计数；结构化 tracing 可测试，保留后续 Prometheus 接口 | 无 |
| TEST-01 | 未开始 | 精度回归 harness | 固定输入/参考输出、余弦相似度阈值、可复用 backend runner；mock 与 OpenVINO 进入通用 CI，硬件后端复用同一格式 | RT-02 |
| TEST-02 | 未开始 | OpenVINO CPU 真实 CI | 安装/缓存 OpenVINO runtime，启用 backend feature，执行真实模型 load → infer → compare，并对 feature path clippy | SYS-01、TEST-01 |
| TEST-03 | 未开始 | 补齐模型/码流 fuzz target | 除现有 config/C ABI 外，覆盖媒体码流/模型元数据等不可信解析面；CI 至少执行 `cargo fuzz check` | MEDIA-01 |
| DEMO-01 | 未开始 | 无硬件多路流多算法综合 demo | `mock://` 多路输入经 decode/resize/inference/track/osd/push 跑通，CLI 集成测试验证，并记录 planned copy count | APP-01、MEDIA-02 |
| DOC-01 | 未开始 | 最终文档与状态收敛 | README/user guide/design 与实际字段、feature、示例、限制一致；删除“已完成”但无实现的陈述 | 其他软件任务 |

## E. 需要真实硬件或自托管 runner 的最终验收

这些任务可以先提交 runner workflow、测试 harness 和报告模板；没有对应设备时状态保持
`外部阻塞`，不得用 stub 结果冒充实机验收。

| ID | 状态 | 独立 PR 范围 | 验收条件 | 依赖/外部资源 |
|---|---|---|---|---|
| HW-01 | 外部阻塞 | RK3588 三核 RKNN 验收 | core0/1/2/auto 的利用率与吞吐报告；动态 shape、量化、dma-buf zero-copy；对比 staging copy count | SCH-02、MEM-02、RK3588 + RKNN SDK |
| HW-02 | 外部阻塞 | NVIDIA TensorRT 验收 | 真实 engine 的 fp32/fp16/int8 精度与吞吐；device_id/多 GPU；CUDA ptr zero-copy | RT-02、MEM-02、NVIDIA GPU + CUDA/TensorRT |
| HW-03 | 外部阻塞 | Sophon Host/SoC 验收 | BM PCIe 与 SE/SoC 的 bmodel 推理、精度、device/core 选择和 device-memory 路径 | RT-02、MEM-02、BM 卡/SE 设备 + SDK |
| HW-04 | 外部阻塞 | 硬解到推理/推流端到端验收 | RTSP → 硬解 → 前处理 → YOLO → track → OSD → RTMP；零拷贝与 CPU staging 的拷贝次数/吞吐对比 | MEDIA-02、DEMO-01、对应 codec/NPU/GPU 硬件 |

## 推荐实施顺序

1. `CFG-02` → `CFG-03` → `CFG-04` → `CFG-05`，先关闭当前明确的加载期校验和 schema 缺口。
2. `CFG-01`、`CFG-06` 至 `CFG-09`、`HOT-01`，再完成配置契约；`HOT-02` 在运行时生命周期稳定后实施。
3. `APP-01`、`MEDIA-01`、`STREAM-01`、`STREAM-02`，打通无硬件的真实软件媒体/协议路径。
4. `SYS-*`、`CORE-01`、`RT-*`、`SCH-*`、`MEM-*`，完成后端分层、调度和真实 zero-copy 软件基础。
5. `OBS-*`、`TEST-*`、`ELEM-*`、`DEMO-*`、`CAPI-01`，完成质量与交付面。
6. 最后在自托管 runner 上执行 `HW-*`，并以实测报告关闭外部阻塞项。
