# avcodec-rs：面向外部 Rust SDK 集成的能力缺口

本文基于 pinned checkout：

```text
/home/ubuntu/.cargo/git/checkouts/avcodec-rs-develop-16e024684163e26c/621a708
```

对应 revision 为 `621a708`。本文中的“现有 API”仅指该 checkout 中可核对的代码；“建议 API”均明确标为 proposed capability，不表示当前已经存在。

> 更新（2026-07）：依赖已升到 upstream `main` HEAD `3e61b5b`。本文五个 gap 在新 revision 上的
> 复核结论与「完成 MEDIA-01 真实视频路径仍缺的能力」见
> [`avcodec-rs-media01-requirements.md`](avcodec-rs-media01-requirements.md)：Gap 2/3/5 已关闭，
> Gap 1/4 仅部分关闭（新增 native-free H.264 **解码**，仍缺 native-free 视频**编码**与完整 packet 契约）。

## 1. 背景/目标

`dyun-gu-dev` 作为外部 Rust integrator，需要在稳定、可 feature-gated 的 SDK 边界上驱动真实媒体路径，并在 CI 中验证这些路径。目标环境不应要求外部服务器、硬件、native SDK、系统库、native build tools 或构建期间下载并编译额外 native artifact。

我们当前实际使用 `avcodec-core-model` 的 `RegistryBuilder`、`Decoder`、`Encoder`、`ImageProcessor` 与 `Poll` 模型，组装 `avcodec-backend-jpeg` 和 `avcodec-backend-zune`。JPEG/MJPEG 是可落地的起点，但不足以覆盖常见视频流。

## 2. 现状（可用的部分）

- `crates/core/avcodec-core-model` 提供 Sans-I/O 的 codec/image model；`RegistryBuilder::new().with_backend(...).build()` 可显式组装 backend。
- `crates/backend/avcodec-backend-jpeg/Cargo.toml` 的 package name 是 `avcodec-backend-jpeg`，依赖 `jpeg-decoder` 与 `jpeg-encoder`；其 `BACKEND` 同时报告 JPEG/MJPEG 的 decode/encode 能力，并支持 host `Gray8`、`Rgb24`、`Rgba`。
- `crates/backend/avcodec-backend-zune/Cargo.toml` 的 package name 是 `avcodec-backend-zune`，依赖 `zune-core`、`zune-jpeg`、`zune-png`、`zune-bmp`、`image-webp` 与 `ab_glyph`。其 `BACKEND` 提供 host packed image processing（包括 resize、CSC、OSD 等）以及 JPEG/PNG/WebP/BMP still-image decode；它不报告 video codec capability。
- `crates/core/avcodec-core-model/src/image.rs` 已定义 `ImageInfo::Yuv420p`、`Yuv422p`、`Yuv444p`、`Nv12`、`Nv21` 等 planar/semi-planar formats，`Image::plane_count`、`ImagePlane`、`HostI420Planes`、`HostNv12Planes` 等模型也存在。
- `crates/core/avcodec-core-model/src/buffer.rs` 提供 `BufferHandle::stage_to_host` 和 `register_stage_to_host_hook`，能表达不同 `MemoryDomain` 的显式 staging；这不是一个统一的 host conversion facade。
- SDK crate `crates/sdk/avcodec/Cargo.toml` 提供 `jpeg`、`zune`、`openh264`、`libvpx`、`dav1d`、`svtav1`、`ffmpeg`、`x264`、`x265` 等 feature。`crates/sdk/avcodec/src/builtins.rs` 的 `default_registry_builder() -> RegistryBuilder` 按 feature 把所有启用的 builtin backend 加入 registry。
- `AvError` 已有结构化的 `AvErrorKind`、`AvErrorDetail`、`kind()`、`detail()` 与 `as_code()`；因此这里不是“没有错误分类”，而是面向外部 telemetry/诊断的上下文仍不完整。

## 3. 缺失能力清单

### Gap 1：没有可保证 native-free 的 SOFTWARE VIDEO codec

**当前不足。** 当前可确认的纯 Rust、无系统 codec runtime 要求路径是 `avcodec-backend-jpeg`（JPEG/MJPEG）和 `avcodec-backend-zune`（still-image/image processing），而不是 H.264、H.265、VP8、VP9 或 AV1 video codec。

checkout 中的 video backend package/feature 证据包括：

| package | SDK feature | 当前代码证据 |
| --- | --- | --- |
| `avcodec-backend-openh264` | `openh264` | `src/lib.rs` 中的 `openh264_library()` 调用 `shiguredo_openh264::Openh264Library::load("libopenh264.so.7"/"libopenh264.so")`；缺少 runtime library 时 probe 为 unavailable |
| `avcodec-backend-libvpx` | `libvpx` | 依赖 `shiguredo_libvpx = "2026.1.0"`，报告 `CodecId::Vp9` |
| `avcodec-backend-dav1d` | `dav1d` | 依赖 `shiguredo_dav1d = "2026.1.0"`，报告 `CodecId::Av1` decode |
| `avcodec-backend-svtav1` | `svtav1` | 依赖 optional `shiguredo_svt_av1 = "2026.1.0"`，feature `sdk` 才启用该依赖，报告 `CodecId::Av1` |
| `avcodec-backend-ffmpeg` | `ffmpeg` | 依赖 `avcodec-codec-ffmpeg`；后者有 `build = "build.rs"` 且依赖 `ffmpeg-sys-next` |
| `avcodec-backend-x264` / `avcodec-backend-x265` | `x264` / `x265` | 通过 `avcodec-codec-x264` / `avcodec-codec-x265` 接入；本 pinned checkout 的 manifest 未展示其底层 runtime 依赖，native-free 行为不能据此假定 |

因此，不能向外部 integrator 承诺 `software-default` 是无 native 依赖的 video profile。至少 `openh264` 明确依赖运行时动态库，`ffmpeg` 明确有 native build path；其余 `shiguredo_*` 依赖的具体 artifact 获取方式应在上游 release contract 中明确，而不应由 integrator 猜测。

这会阻碍需要 H.264/H.265/VP8/VP9/AV1 的外部 SDK：只能退回 JPEG 或自行引入未受上游统一保证的依赖，CI 也无法用同一个 dependency-light matrix 稳定验证。

**建议的 capability（proposed）。**

1. 至少提供一个明确标注为 native-free 的 software video decoder，例如 `H264`、`VP8`、`VP9` 或 `AV1`；理想情况下同时提供 encoder。
2. 为该 backend 提供 feature（例如 `software-video-native-free`，名称仅为建议）和明确的 Cargo/build contract：不链接系统库、不运行下载型 build script、不要求 native SDK。
3. 在 capability/probe API 中区分“编译进来但 runtime unavailable”和“可在 dependency-free CI 中实际运行”。

**优先级：P0。**

### Gap 2：没有官方的 native-free software registry guarantee

**当前不足。** `avcodec-core-model` 的 `RegistryBuilder` 只保存显式传入的 `&'static dyn BackendFactory`；core/model 本身不 bundle backend。SDK 的 `default_registry_builder()` 则按 feature 注册 backend，而 `software-default` 的定义是：

```toml
software-default = [
    "jpeg", "libyuv", "zune", "video-device",
    "x264", "x265", "openh264", "libvpx", "dav1d", "svtav1"
]
```

这个 feature 名称看起来像 software profile，但其组成包含 `openh264`、`x264`、`x265` 等可能需要 native runtime/artifact 的 backend，并且还包含 `video-device`。它没有一个由 API 或文档保证“无 native 依赖、可在干净 CI 构建和运行”的 preset。

**为什么阻碍外部 SDK/CI。** 外部 integrator 只能手写：

```rust
RegistryBuilder::new()
    .with_backend(&backend_jpeg::BACKEND)
    .with_backend(&backend_zune::BACKEND)
    .build()
```

并自行审计每个 backend；使用 `default_registry_builder()` 又会把 feature 开启的其它 backend 一并带入。这使 Cargo feature、CI 依赖和运行时 probe 的语义不稳定。

**建议的 capability（proposed）。** 提供一个有明确文档和测试契约的 preset，例如：

```rust
// proposed API；当前 checkout 中不存在
pub fn native_free_software_registry_builder() -> RegistryBuilder;
```

或等价 feature `native-free-software`。其契约应至少包括：

- 只注册已经审计为 native-free 的 backend；
- 不依赖系统库、硬件或 build-time download；
- 在干净 Linux/macOS/Windows（按上游支持矩阵）运行最小 encode/decode/process smoke test；
- 暴露稳定的 `backend_ids()`/capability 清单；
- 对不能满足的 codec 返回结构化 `Unsupported`/probe reason，而不是隐式选择另一个 backend。

**优先级：P0。**

### Gap 3：跨 packed/planar `Image` 到 host SDK 的通用 conversion facade 不足

**当前不足。** core 已有格式和 plane model：`ImageInfo::plane_count()`、`ImagePlane`、`HostI420Planes`、`HostNv12Planes` 以及 `Image::plane_host_bytes()` 等分散 API；`BufferHandle::stage_to_host()` 还需要按 `MemoryDomain` 注册 hook。与此同时，当前 backend 的公开能力并不统一：`avcodec-backend-jpeg` 主要支持 packed `Gray8`/`Rgb24`/`Rgba`，`avcodec-backend-zune` 的 processor 也明确是 packed host formats，而 `avcodec-backend-dav1d` 的实现会处理 I420 planes。

MEDIA-01 的 bridge 因而必须自行处理 plane 数量、subsampling、每 plane stride、packed bytes-per-pixel、host staging 和 output shape。这个工作在 JPEG 路径已完成，但不是一个可复用的、对外稳定的 `Image` ↔ host frame helper。`Image` 类型本身也没有一个统一的“导出所有 planes 及其有效 row bytes/stride/shape”的单一 facade。

**为什么阻碍外部 SDK/CI。** 新 integrator 接入 YUV video backend 时很容易把 plane stride 当作 width、丢失 chroma subsampling 或错误地复制 padded rows；这些错误通常只能在特定 codec/尺寸下发现，难以写成 backend-independent CI。

**建议的 capability（proposed）。**

```rust
// proposed API；当前 checkout 中不存在
pub struct HostImageView<'a> {
    pub format: ImageInfo,
    pub width: usize,
    pub height: usize,
    pub planes: &'a [HostPlaneRef<'a>],
}

pub fn image_to_host_view(image: &Image) -> AvResult<HostImageView<'_>>;
pub fn copy_image_to_host(image: &Image) -> AvResult<HostImage>;
```

建议 API 应明确每 plane 的有效高度、row bytes、stride、offset、sample type/layout，并提供从 host packed/planar buffers 构造 `Image` 的对称 helper；不能把 staging copy 隐式混入 zero-copy 路径。

**优先级：P1。**

### Gap 4：video packet 的 extradata/parameter-set 与时间戳契约不够集中

**当前不足。** `avcodec-core-model` 已有 `Packet`、`PacketFlags`、`TimeBase`、PTS/DTS 等 codec model；音频配置在 `crates/core/avcodec-core-model/src/audio.rs` 中有 `AudioDecoderConfig::extra_data`，但在本次核对的 video `Image`/`Packet`/`DecoderConfig`/`EncoderConfig` public model 中，没有找到一个等价且统一的 video codec extradata/parameter-set metadata API，能让外部 integrator 以稳定方式读取或携带 H.264/H.265/AV1 configuration records。当前 backend 是否把 parameter sets 放在 packet payload、backend-specific state 或其它字段，也没有一个跨 backend contract。

这不是说当前 codec model 没有 PTS/DTS：它有相关字段；缺口是外部 integrator 缺少统一、文档化的“packet metadata preservation” contract，尤其在 packet bridge、flush 和 reordering 时。

**为什么阻碍外部 SDK/CI。** 将 decoded/encoded packet 接入 RTSP/RTMP/WebRTC 等协议时，integrator 必须知道 extradata、keyframe flags、PTS/DTS/timebase 的来源和生命周期。若只能依赖 backend-specific 观察，协议启动和 CI fixture 会出现 codec-specific 分支。

**建议的 capability（proposed）。**

```rust
// proposed API；当前 checkout 中不存在
pub struct CodecParameters {
    pub codec: CodecId,
    pub extradata: Option<BufferSlice>,
}

pub trait PacketMetadata {
    fn codec_parameters(&self) -> Option<&CodecParameters>;
    fn pts(&self) -> Option<i64>;
    fn dts(&self) -> Option<i64>;
    fn time_base(&self) -> TimeBase;
}
```

具体命名可以不同，但应规定 parameter-set/extradata、keyframe flags、PTS/DTS/timebase 在 `submit`、`poll`、`flush` 和 reset 后如何保留，并提供无 native 的 round-trip fixture。

**优先级：P1。**

### Gap 5：评估 `AvError` 的上下文丰富度

**现状判断。** `AvError` 的定义位于 `crates/core/avcodec-core-model/src/error.rs`：

```rust
pub enum AvError {
    InvalidArgument,
    Unsupported,
    Again,
    EndOfStream,
    BufferDomainMismatch,
    NotInitialized,
    QueueFull,
    BackendFailure,
    BackendMessage(String),
    InvalidState,
    CycleDetected,
    DeviceLost,
    OutOfMemory,
    Classified { kind: AvErrorKind, detail: AvErrorDetail },
    ExternalError(i32),
}
```

它能提供 kind/detail/code，且 `AvErrorDetail` 覆盖 request mismatch、backend selection、dependency failure 等类别；这足以支持基本的 polling/EOS 和错误分类映射。因此“缺少任何错误分类”不是已证实的 gap。

但 `AvError` 本身没有结构化的 backend、codec、operation、frame/packet、stream/track、source/destination format 或具体 dimensions 字段。`BackendMessage(String)` 与 `ExternalError(i32)` 也无法稳定承载这些上下文。外部 integrator 在映射到 `dg-core::Error::Media`、日志和 telemetry 时，仍需把上下文附加在调用方字符串中。

**建议的 capability（proposed）。** 保持现有 enum 兼容性的前提下，增加可选结构化 context，例如：

```rust
// proposed API；当前 checkout 中不存在
pub struct AvErrorContext {
    pub backend_id: Option<&'static str>,
    pub codec: Option<CodecId>,
    pub operation: Option<Operation>,
    pub frame_index: Option<u64>,
    pub packet_index: Option<u64>,
    pub source_format: Option<ImageInfo>,
    pub destination_format: Option<ImageInfo>,
}
```

也可以提供 `AvError::with_context(...)`、`backend_id()`、`operation()` 等访问器，但应保留 `kind()`、`detail()`、`as_code()`。`Again` 和 `EndOfStream` 仍应继续表达 polling/EOS semantics，而不是被外部 integrator 当作 hard failure。

**优先级：P2。**

## 4. 验收建议

建议上游为每项 capability 提供可复制的 Rust example 和 CI test：

1. `native-free-software` profile 在无系统 codec library、无硬件、无下载型 build script 的干净环境中执行；
2. 至少一个纯 Rust video codec 完成真实 bitstream decode；若提供 encode，则执行 encode→decode round-trip；
3. `RegistryBuilder`/preset 的 `backend_ids()`、capability probe 和 unavailable reason 有断言；
4. packed `Gray8`/`Rgb24`/`Rgba`、planar `Yuv420p`、semi-planar `Nv12` 的 host conversion test 覆盖 padded stride、subsampling 和 staging；
5. packet fixture 断言 codec、extradata/parameter sets、keyframe flags、PTS/DTS/timebase 在 `submit`/`poll`/`flush` 后保留；
6. error fixture 断言 `AvErrorKind`/`AvErrorDetail` 以及 backend/operation/format context 可供外部 SDK 稳定读取。
