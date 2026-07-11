# avcodec-rs：完成 MEDIA-01 真实视频路径仍需的上游能力

> 复核基线：upstream `main` HEAD
> `3e61b5b39366dec91a6e4653787773dbcd8dfd6d`
> （相对本仓库原 pin `621a708`，`main` 增量为 `git log --oneline 621a708..3e61b5b` 共 215 个提交）。
>
> 本文是 [`avcodec-rs-gaps.md`](avcodec-rs-gaps.md) 的后续复核与需求收敛：`avcodec-rs-gaps.md`
> 记录的是 pinned rev `621a708` 上的能力缺口；本文在把依赖更新到 `main` HEAD 后，重新核对
> 那五个 gap 的落地情况，并把「要真正开发 MEDIA-01 的真实软件视频路径仍然缺少」的能力
> 收敛成面向上游的需求。文中「现有 API」仅指 `3e61b5b` checkout 中可核对的代码；「建议 API」
> 均标注为 proposed，不表示当前已经存在。

## 1. 背景/目标

`dyun-gu-dev` 的 MEDIA-01（[design-remaining-tasks.md](../design-remaining-tasks.md)、[design.md §9.1](../design.md)）
要求 `dg-media` 通过 `avcodec` feature 用 `RegistryBuilder` 驱动 `Decoder`/`Encoder`/`ImageProcessor`，
并在 **无系统库、无硬件、无下载型 build script、无 native SDK** 的干净 CI 中，用真实码流覆盖
software codec 路径。当前 `dg-media-avcodec` 已把依赖从 `621a708` 更新到 `main` HEAD `3e61b5b`，
适配层仍稳定编译并通过既有 JPEG/MJPEG + resize/OSD 测试。

本文回答两个问题：

1. 更新到 `main` 之后，`avcodec-rs` 是否已经满足 MEDIA-01 的**真实视频 decode/encode**要求？
2. 若不满足，作为外部 integrator，我们还需要上游提供哪些能力（不 fork 上游，只提交需求）。

**结论：仍不满足。** `main` 已经补齐 native-free registry preset、host image conversion facade、
结构化 `AvError` context，并新增了**纯 Rust、native-free 的 H.264 软解码器**；但**没有任何
native-free 的软件视频编码器**，且跨 backend 的 packet/extradata/timebase 契约只完成一半。
因此 MEDIA-01 目前最多只能落地「真实 H.264 软解码 → I420」这一半，无法完成 decode/encode 闭环。

## 2. 相对 `621a708` 的 gap 复核

对 [`avcodec-rs-gaps.md`](avcodec-rs-gaps.md) 五个 gap 在 `3e61b5b` 上的复核：

| Gap | 原优先级 | `3e61b5b` 状态 | 说明 |
| --- | --- | --- | --- |
| Gap 1：native-free software VIDEO codec | P0 | **部分关闭** | 新增纯 Rust、native-free 的 H.264 **解码器**；但仍无 native-free 视频**编码器**，也无 native-free VP8/VP9/AV1。 |
| Gap 2：native-free software registry guarantee | P0 | **已关闭** | 新增 `native_free_software_registry_builder()` 与 `native-free-software` feature，并有测试/校验契约。 |
| Gap 3：packed/planar `Image` ↔ host conversion facade | P1 | **已关闭** | 新增 `HostImageView`/`HostPlaneGeometry`/`image_to_host_view`/`copy_image_to_host` 等稳定 facade。 |
| Gap 4：video packet extradata/parameter-set 与时间戳契约 | P1 | **部分关闭** | 新增 `CodecParameters`、`Packet` 显式 codec/bitstream_format/PTS/DTS/flags 与 `DecoderConfig.parameters`；但 encoder 侧 extradata 暴露、packet 级 timebase、统一 `PacketMetadata` 契约仍缺。 |
| Gap 5：`AvError` 上下文丰富度 | P2 | **已关闭** | 新增 `AvErrorContext`/`AvOperation`/`with_context()`/`context()`，且保留 `kind()`/`detail()`/`as_code()`。 |

### 2.1 已关闭 gap 的落地证据（`3e61b5b`）

- **Gap 2 — native-free registry preset。** `crates/sdk/avcodec/Cargo.toml` 新增
  `native-free-software = ["jpeg", "zune", "rust-h264"]`；`crates/sdk/avcodec/src/builtins.rs`
  提供 `pub fn native_free_software_registry_builder() -> RegistryBuilder`、常量
  `NATIVE_FREE_SOFTWARE_BACKEND_IDS = &["jpeg", "zune", "rust-h264"]`，并在文档中显式声明
  它「never registers native-runtime, prebuilt-download, device, or hardware backends」。
  `default_registry_builder()` 保留但注明「not a native-free guarantee」。
- **Gap 3 — host image conversion facade。** `crates/core/avcodec-core-model/src/image.rs`
  新增 `HostImageView<'a>`、`HostPlaneGeometry<'a>`（含 `offset`/`stride`/`width_px`/`height_px`/
  `effective_row_bytes`）、`image_to_host_view()`（非 host 域返回 `BufferDomainMismatch`，不隐式拷贝）、
  `copy_image_to_host()`（显式 `stage_to_host`）、`host_image_to_packed()`、`from_host_image()`，
  以及 `host_i420_planes()`/`host_nv12_planes()`；覆盖 padded stride、subsampling、staging、奇数尺寸。
- **Gap 5 — 结构化 `AvError` context。** `crates/core/avcodec-core-model/src/error.rs` 新增
  `AvErrorContext { backend_id, codec, operation, frame_index, packet_index, source_format,
  destination_format, width, height }`、稳定枚举 `AvOperation`、`AvError::with_context()` 与
  `context()`，并保持 `kind()`/`detail()`/`as_code()` 兼容（`Again`/`EndOfStream` 语义不变）。

这三项已可在 MEDIA-01 的 bridge / 错误归一化 / registry 组装中直接采用，不再需要 integrator 侧自造。

## 3. 仍缺失的能力清单（面向上游的需求）

以下需求在 `3e61b5b` 上仍未满足，是完成 MEDIA-01「真实软件视频 decode/encode 闭环」的直接阻塞项。

### Req A：native-free software VIDEO **encoder**（对应 Gap 1 未关闭部分）

**当前不足。** `main` 已提供纯 Rust、native-free 的 H.264 **解码器**：
`crates/backend/avcodec-backend-rust-h264`（依赖 `rust_h264 = "0.4.0"`，Annex-B 输入 → I420 输出，
`static BACKEND: RustH264Backend`），SDK feature `rust-h264`，并有真实 bitstream fixture
（`tests/fixtures/smoke.h264`）与集成测试。其 `capability.toml` 明确：

```toml
backend = "rust-h264"
decode = true
encode = false
decode_codecs = ["H264"]
encode_codecs = []
```

但 native-free preset（`jpeg`/`zune`/`rust-h264`）中**没有任何视频编码器**：仍需 native runtime
或下载型 build 的 `openh264`（`shiguredo_openh264::Openh264Library::load("libopenh264.so.7")`）、
`x264`/`x265`、`libvpx`、`svtav1`、`ffmpeg`（`avcodec-codec-ffmpeg` 带 `build.rs` + `ffmpeg-sys-next`）
才能做视频编码。因此 MEDIA-01 的 `EncodeCore`（`create_encoder` → `submit_frame` → `poll_packet`）
无法在 native-free 环境里跑真实视频编码，只能落地 decode 半程或退回 JPEG。

**建议的 capability（proposed）。**

1. 至少提供一个明确标注为 native-free、纯 Rust 的软件视频 **encoder**（首选 `H264`，与既有
   `rust-h264` decoder 对称），使 `native-free-software` preset 能完成 encode。
2. 该 encoder 的 `capability.toml` 报告 `encode = true` / `encode_codecs = ["H264", ...]`，
   并沿用 `rust-h264` 的 Cargo/build 契约：不链接系统库、不跑下载型 build script、不要求 native SDK。
3. 提供 native-free 的 **encode → decode round-trip** fixture/test（复用现有 validation profile），
   证明 `submit_frame`/`poll_packet`/`flush` 的真实码流闭环。
4. 若短期只能提供 decode，请在 release contract 中明确「native-free 仅覆盖 H.264 decode」，
   以便 integrator 据此把 MEDIA-01 拆成 decode-only 中间里程碑。

**优先级：P0。**（不满足则 MEDIA-01 的 encoder 验收无法在通用 CI 中完成。）

### Req B：native-free 视频 codec 覆盖面（对应 Gap 1 的横向扩展）

**当前不足。** native-free 路径目前只有 H.264 decode。VP8/VP9/AV1 仍分别落在
`libvpx`/`dav1d`/`svtav1` 等 `shiguredo_*`（native runtime）或 `ffmpeg`（native build）后端上，
不在 native-free 保证内。

**建议的 capability（proposed）。** 若 MEDIA-01 / 下游 demo 需要 H.264 以外的 codec，请说明
是否计划提供 native-free 的 VP8/VP9/AV1 decode（及可选 encode），或明确这些 codec 只在
native/hardware profile 支持。integrator 需要一个稳定的「哪些 codec 属于 native-free 保证」清单，
而不是从各 backend manifest 逐个推断。

**优先级：P1。**（H.264 足以打通首期端到端；多 codec 覆盖影响 DEMO/后续验收广度。）

### Req C：完成跨 backend 的 packet / extradata / timebase 契约（对应 Gap 4 未关闭部分）

**当前不足。** `main` 已新增 `crates/core/avcodec-core-model/src/codec_params.rs` 的
`CodecParameters { codec, extradata, bitstream_format }`、`DecoderConfig.parameters`
（`crates/core/avcodec-core-model/src/traits.rs`）、显式 `Packet { stream_index, codec,
bitstream_format, pts, dts, flags, data }`（`packet.rs`），并在 `request.rs` 中对
`H264Avcc`/`H265Hvcc` 等 out-of-band 格式要求 extradata。`rust-h264` backend 也用 FIFO
保留 PTS/DTS 并经 flush 透传。

但对 integrator 稳定接入 RTSP/RTMP/WebRTC（STREAM-01/02、MEDIA-02）而言仍缺：

- `EncoderConfig` 没有 `CodecParameters` / extradata 字段，encoder 生成的 parameter set
  （SPS/PPS/VPS、AV1 config record）**没有统一的输出契约**；
- `TimeBase` 仍挂在 `DecoderConfig`/`EncoderConfig` 上，**未随每个 `Packet` 携带**；
- 没有统一的 `PacketMetadata`-风格 accessor trait，也没有文档规定 extradata/keyframe flags/
  PTS/DTS/timebase 在 `submit`/`poll`/`flush`/`reset` 之后如何保留；
- native-free backend 是 decoder-only，encoder 侧的 parameter-set 保真无法在 native-free CI 中示范。

**建议的 capability（proposed）。** 在保持现有类型兼容的前提下：

```rust
// proposed API；当前 checkout 中不存在
impl EncoderConfig {
    pub fn with_parameters(self, params: CodecParameters) -> Self;
}

pub trait PacketMetadata {
    fn codec_parameters(&self) -> Option<&CodecParameters>; // encoder 输出的 SPS/PPS/extradata
    fn time_base(&self) -> TimeBase;                        // packet 级 timebase
    fn pts(&self) -> Option<i64>;
    fn dts(&self) -> Option<i64>;
    fn is_keyframe(&self) -> bool;
}
```

并提供 native-free 的 round-trip fixture，断言 codec、extradata/parameter sets、keyframe flags、
PTS/DTS/timebase 在 `submit`/`poll`/`flush`/`reset` 后保留（与 Req A 的 encoder 一并交付最理想）。

**优先级：P1。**（decode-only 路径已可用；encoder 侧 parameter-set 与协议对接依赖此项。）

## 4. 验收建议

沿用 [`avcodec-rs-gaps.md` §4](avcodec-rs-gaps.md) 的思路，针对本文新增/未关闭项：

1. `native-free-software` preset 在无系统 codec library、无硬件、无下载型 build script 的干净环境中，
   完成至少一次真实 H.264 **encode → decode** round-trip（Req A）。
2. 若提供 H.264 以外 native-free codec，给出对应 decode（及可选 encode）真实码流 test，
   并在文档中给出 native-free codec 清单（Req B）。
3. packet fixture 断言 encoder 输出的 codec、extradata/parameter sets、keyframe flags、
   PTS/DTS/timebase（packet 级）在 `submit`/`poll`/`flush`/`reset` 后保留（Req C）。

## 5. 对 `dyun-gu-dev` 的影响

- 依赖已更新到 `main` HEAD `3e61b5b`；`dg-media` 的 registry/错误归一化/bridge 可开始采用已关闭
  gap（native-free preset、`HostImageView`、`AvErrorContext`）。
- MEDIA-01 的真实视频路径可先落地「H.264 native-free 软解码 → I420」中间里程碑（Req A 的 decode 半程已具备）。
- 完整 MEDIA-01（decode/encode 闭环 + 协议可用的 packet 契约）仍**外部阻塞**于 Req A（P0）与 Req C（P1）；
  在上游提供 native-free 视频 encoder 之前，encoder 验收不得用 JPEG 或 native backend 冒充。
