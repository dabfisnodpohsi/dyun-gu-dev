# cheetah-media-server-rs：`cheetah-connector` 复核与 STREAM-01 能力状态

> 本文是对 [`cheetah-media-server-rs-gaps.md`](cheetah-media-server-rs-gaps.md)
> 及此前 connector 复核的更新审计。
> 原文基于 pinned rev `182621c`，此前复核基于上游 `206fc11`；本次基于上游
> `48cc74e` 重新核对实际代码与构建结果，确认上游已经补齐 STREAM-01 所需的
> RTSP/HTTP-FLV pull、RTMP/WebRTC push 四项能力，并提供可选的 socket-free
> loopback 路径。仅 RTMP→HTTP-FLV wire metadata 的逐字段保真仍不完整，该问题属于
> STREAM-02。

## 0. 复核基线

- `dyun-gu-dev` 当前锁定 rev（`crates/dg-stream-cheetah/Cargo.toml`）：
  `182621c393eff754660a445d174a61e41622c660`。
  本次未修改该依赖；升级仍需由 `dyun-gu-dev` 后续变更完成。
- 本文复核的上游 HEAD：
  `48cc74e050c4bf20dc7d8b111b83f2ab00b7999c`
  （Merge pull request #84 from ChungTak/devin/1783755269-webrtc-push）。
- 自上次 `206fc11` 复核以来的关键落地：
  - PR #82：`plan2 residual gaps S1/S4/S5/S6/S7`，覆盖选项透传、就绪信号、
    socket-free loopback、metadata/error 相关残余缺口。
  - PR #83：RTSP pull adapter。
  - PR #84：WebRTC push adapter。
- 本次构建验证：
  在上游 `48cc74e` 执行
  `cargo build -p cheetah-connector --features full` 成功。
- 以下“现有 API/行为”均指上游 HEAD `48cc74e` 可核对的代码；状态以实际
  `open_pull`/`open_push` 分支和测试为准，而非仅依据 capability 声明。

## 1. 结论摘要

| 要求 | HEAD 现状 | STREAM-01 是否阻塞 |
|---|---|---|
| RTSP pull | **已接线**：`open_pull(Protocol::Rtsp, ...)` 调用 RTSP adapter，不再返回 `UnsupportedProtocol` | 否 |
| HTTP-FLV pull | **已实现**：流式 `SubscriberSource` 已接入 connector | 否 |
| RTMP push | **已实现**：`open_push(Protocol::Rtmp, ...)` 调用 RTMP adapter | 否 |
| WebRTC push | **已接线**：`open_push(Protocol::WebRtc, ...)` 调用 WHIP/WebRTC adapter，不再返回 `UnsupportedProtocol` | 否 |
| capability matrix | **一致**：四个必需 `(protocol, direction)` 均由 `supports()` 声明并有实际 adapter 分支 | 否 |
| socket-free loopback | **已提供**：`LoopbackLayer::EngineOnlyBypassWire` 直接使用 engine `StreamManager` | 否 |
| metadata 保真 | **部分**：RTMP→HTTP-FLV wire 路径仍丢失/规范化部分 `AVFrame` 字段 | 仅 STREAM-02 |

**一句话**：上游 `48cc74e` 已满足 STREAM-01 的完整能力矩阵：
**RTSP pull + HTTP-FLV pull + RTMP push + WebRTC push + socket-free local loopback**。
`dyun-gu-dev` 侧只需将 `dg-stream-cheetah` 的依赖 rev bump 到包含
`cheetah-connector` 及上述提交的版本，并实现相应 facade 集成测试。STREAM-02
仍受 R7 的 metadata 保真限制影响。

## 2. 已可直接消费的能力

以下能力已在 HEAD 落地：

- 高层 facade：`RuntimeConnector`、`EngineConnector`、`ConnectorBuilder`
  （`crates/sdk/cheetah-connector/src/connector.rs`、
  `.../src/engine_bootstrap.rs`）。
- 句柄：`PullHandle`（`tracks()`/`recv()`/`close()`，并实现
  `SubscriberSource`）、`PushHandle`（`update_tracks()`/`push_frame()`/
  `take_keyframe_requests()`/`wait_ready()`/`close()`，并实现
  `PublisherSink`）、`LoopbackPair`（`.../src/handles.rs`）。
- RTSP pull：`pull::rtsp::open_rtsp_pull`。
- 流式 HTTP-FLV pull：`pull::http_flv::open_http_flv_pull` 及底层
  `open_http_flv_subscriber`，含 reconnect/backoff、bounded queue、cancel。
- RTMP push：`push::rtmp::open_rtmp_push`。
- WebRTC push：`push::webrtc::open_webrtc_push`，通过 WHIP signaling 和
  WebRTC driver 生成可写的 `PublisherSink`。
- loopback：`open_in_memory_loopback`。默认 RTMP→HTTP-FLV 拓扑仍是
  `ProtocolFraming`（localhost TCP），但可请求 `EngineOnlyBypassWire` 获得
  无 socket 的 engine-only 路径；WebRTC 另有 media fixture。
- typed error：`ConnectorError`，带 `protocol()`/`retryable()`/`source()`；
  泛化 `SdkError::Unavailable` 不再臆测为 RTMP。
- 能力矩阵：`Protocol`/`Direction`/`supports()`，按 feature 且按实际 adapter
  接线状态判定。
- feature 门控：`rtsp`/`http-flv`/`rtmp`/`webrtc`/`loopback`/`full`。

## 3. R1..R8 复核结果

### R1：RTSP pull adapter —— **FULLY IMPLEMENTED**

上游已将 RTSP pull 接入高层 connector：

- `crates/sdk/cheetah-connector/src/pull/rtsp.rs:21-60`：
  `open_rtsp_pull(engine, url, options)` 解析 stream key/source peer，
  将 `ConnectorPullOptions.subscriber` 和 cancellation 传入
  `cheetah_rtsp_module::pull::open_rtsp_pull`，并包装为 `PullHandle`。
- `crates/sdk/cheetah-connector/src/connector.rs:121-124`：
  `EngineConnector::open_pull` 的 `Protocol::Rtsp` 分支实际调用
  `crate::pull::rtsp::open_rtsp_pull(...)`，不再返回
  `ConnectorError::UnsupportedProtocol`。
- `crates/sdk/cheetah-connector/tests/rtsp_pull.rs:163-166,197-204`：
  通过公开 connector 打开 RTSP pull，并覆盖关闭、取消及 bounded queue。

**已落地。** STREAM-01 所需的 RTSP `SubscriberSource` 已可通过
`open_pull(Protocol::Rtsp, ...)` 获取。

### R2：WebRTC push adapter —— **FULLY IMPLEMENTED**

上游已将 WebRTC push 接入高层 connector：

- `crates/sdk/cheetah-connector/src/push/webrtc.rs:78-158`：
  `WebRtcPublisherSink` 实现 `PublisherSink`，支持 tracks、frame、close、
  keyframe request 和有界 command/frame buffer。
- `.../src/push/webrtc.rs:163-220`：
  `open_webrtc_push(engine, url, options)` 规范化 WHIP URL、启动 WebRTC
  driver、创建 signaling/background task，并返回 `PushHandle`。
- `crates/sdk/cheetah-connector/src/connector.rs:156-159`：
  `EngineConnector::open_push` 的 `Protocol::WebRtc` 分支实际调用
  `crate::push::webrtc::open_webrtc_push(...)`，不再返回
  `ConnectorError::UnsupportedProtocol`。
- `crates/sdk/cheetah-connector/tests/webrtc_push.rs:358-381`：
  本地 WHIP answerer 验证 `wait_ready()`、track 更新及 H.264 keyframe
  到达对端。

**已落地。** STREAM-01 所需的 WebRTC `PublisherSink` 已可通过
`open_push(Protocol::WebRtc, ...)` 获取。该 adapter 测试使用本地 signaling/
driver peer，不依赖外部服务。

### R3：能力矩阵与实际实现一致 —— **FULLY IMPLEMENTED**

- `crates/sdk/cheetah-connector/src/protocol.rs:35-56`：
  `supports()` 按 feature 且按 adapter wired 状态返回结果：
  - RTSP pull
  - HTTP-FLV pull
  - RTMP push
  - WebRTC push
- `crates/sdk/cheetah-connector/src/connector.rs:115-174`：
  四个声明支持的方向分别进入实际 adapter；其他方向仍返回
  `UnsupportedProtocol`。
- `crates/sdk/cheetah-connector/src/engine_bootstrap.rs:227-264`：
  feature 未启用时返回 `FeatureDisabled`，而不是虚假宣称支持。

**已落地。** `supports()` 与 `open_pull`/`open_push` 的实际行为一致，
此前“声明支持但返回 `UnsupportedProtocol`”的矛盾已关闭。

### R4：connector 层选项透传 —— **FULLY IMPLEMENTED**

- `crates/sdk/cheetah-connector/src/pull/http_flv.rs:25-43`：
  HTTP-FLV reconnect、read limits、buffer size、cancel 均传到底层；
  `SubscriberOptions.queue_capacity` 在未显式指定 buffer 时作为 buffer
  capacity。
- `.../src/pull/rtsp.rs:39-45`：
  RTSP 专用选项与 `SubscriberOptions` 被传到底层 RTSP pull。
- `.../src/push/rtmp.rs:174-198`：
  RTMP command/write queue、read buffer、chunk size、ACK window 等
  `RtmpPushExtras` 被用于构造 driver config。
- `.../src/loopback.rs:133-153`：
  `LoopbackOptions.queue_capacity` 传入 HTTP-FLV buffer 和 RTMP command/write
  queues；`.../src/loopback.rs:190-193` 在 engine-only 路径传入 subscriber。
- 对应覆盖见 `tests/options_passthrough.rs`。

**已落地。** 原先硬编码 HTTP-FLV buffer/read 参数及未使用 loopback queue
的问题已修复。

### R5：`PushHandle::wait_ready()` 真实就绪语义 —— **FULLY IMPLEMENTED**

- `crates/sdk/cheetah-connector/src/handles.rs:107-112,123-136`：
  `PushHandle` 持有 protocol adapter 提供的 `tokio::sync::watch` readiness
  receiver。
- `.../src/handles.rs:177-196`：
  `wait_ready()` 在初始未就绪时等待 channel 变化，并将 channel drop/失败
  映射为 typed `ConnectorError`。
- RTMP 在 publish 状态建立后发送 ready：
  `.../src/push/rtmp.rs:224-231,268-269`。
- WebRTC 在 lifecycle `Connected` 且 media mids 可用后发送 ready：
  `.../src/push/webrtc.rs:279-289,324-355`。
- WebRTC readiness 测试见 `tests/webrtc_push.rs:335-346`；loopback 调用见
  `tests/loopback_layers.rs:95,164`。

**已落地。** 推流侧不再需要依赖 sleep/轮询猜测可写时机。

### R6：socket-free/in-memory loopback —— **FULLY IMPLEMENTED（默认层级有 caveat）**

上游现在同时提供 protocol-framing 和真正不经 socket 的 engine-only 路径：

- `crates/sdk/cheetah-connector/src/loopback.rs:34-69`：
  `open_in_memory_loopback` 根据 `LoopbackLayer` 选择路径。
- `.../src/loopback.rs:169-208`：
  `engine_only_loopback` 直接使用 engine 的 `StreamManager` 打开
  publisher/subscriber，不构造 URL、不连接 TCP socket，返回
  `LoopbackLayer::EngineOnlyBypassWire`。
- `crates/sdk/cheetah-connector/tests/loopback_layers.rs:75-123`：
  engine-only 路径完成 H.264/AAC frame round-trip，证明可用于本地集成测试。

需要明确区分：

- 默认 `LoopbackLayer::ProtocolFraming` 仍通过 service registry 获取端点，
  使用 localhost TCP RTMP/HTTP-FLV（`loopback.rs:72-166`）。
- 严格的“不开 socket”验收应设置
  `LoopbackOptions.preferred_layer = LoopbackLayer::EngineOnlyBypassWire`。

**已落地。** socket-free 路径已具备且有测试；默认 protocol-framing 的行为
已在 `options.rs:145-157` 明确记录。

### R7：RTMP→HTTP-FLV metadata 逐字段保真 —— **PARTIAL（STREAM-02）**

受测路径仍能保留 connector 运行所需及测试覆盖的主要字段，包括：
`track_id`、`media_kind`、`codec`、`format`、`pts`、`dts`、`timebase`、
关键帧标志、payload，以及 track 级 codec/clock rate/sample rate/channels/
extradata。证据见 `tests/metadata_conformance.rs:80-160`。

但 RTMP FLV wire serialization 后由 ingress 重建 `AVFrame`，以下字段仍不
逐字段保真：

- `duration`/`duration_us` 不随 FLV 传输，重建后为 0。
- `origin` 被 ingress 设为 `FrameOrigin::Ingest`。
- `side_data` 不整体保留，ingress 会生成/替换 source timestamp 等数据。
- 音频 flags 被规范化为 `START_OF_AU | END_OF_AU`；视频非关键帧的
  `DISCONTINUITY`、`CORRUPT`、`DROPPABLE`、`GENERATED` 等不保证保留。
- `pts_us`/`dts_us` 由重建 timebase 重新计算。
- track extradata 可能被规范化（例如 H.264 `avcc: None` 变为解析后的
  `Some(...)`）。

证据：

- `crates/foundation/cheetah-codec/src/flv_ingress.rs:247-263,440-455,539-545`
- `crates/sdk/cheetah-connector/src/push/rtmp.rs:120-141`
- `crates/sdk/cheetah-connector/tests/metadata_conformance.rs:82,136-160`

**仍需后续处理。** STREAM-01 主验收不被该项阻塞；STREAM-02 应继续选择：
在 FLV 映射中扩展必要 metadata，或在 connector 契约中明确声明不保真字段并
维持对应 conformance 测试。

### R8：`SdkError → ConnectorError` 不应臆测 RTMP —— **FULLY IMPLEMENTED**

- `crates/sdk/cheetah-connector/src/error.rs:152-165`：
  `SdkError::Unavailable` 不再映射成硬编码的
  `ConnectorError::Connect { protocol: Protocol::Rtmp, ... }`，而是映射为
  `ConnectorError::Internal`，避免错误标注协议。
- 句柄已知协议时继续使用
  `handles.rs:215-230` 的 `map_sdk_error(protocol, operation, err)` 注入
  正确上下文。
- `tests/error_conformance.rs:63-69` 验证泛化 `Unavailable` 结果没有 protocol
  且不可重试。

**已落地。** 随 RTSP/WebRTC adapter 增加，泛化错误映射不会再错误归因到 RTMP。

## 4. 上游验收结果

在 `crates/sdk/cheetah-connector` 内，本次复核确认以下验收项已经满足：

1. `supports()` 与四个实际 `open_pull`/`open_push` 分支一致（R3）。
2. RTSP pull 具备真实 streaming `recv`、取消、关闭和 bounded queue 覆盖；
   adapter 已接线（R1）。
3. WebRTC push 具备真实 `open_push`、WHIP signaling、driver media publish
   路径和 keyframe peer round-trip 测试（R2）。
4. 提供并测试 `EngineOnlyBypassWire` socket-free loopback；默认
   `ProtocolFraming` localhost TCP caveat 已明确标注（R6）。
5. metadata conformance 明确记录 RTMP→HTTP-FLV 的不保真集合；该项仍为
   STREAM-02 的 R7 残余，不阻塞 STREAM-01。
6. `wait_ready()` 使用 RTMP publish/WebRTC connected 的真实 readiness channel，
   并有测试覆盖（R5）。

构建验证：

```text
cargo build -p cheetah-connector --features full
Finished `dev` profile [unoptimized + debuginfo]
```

## 5. `dyun-gu-dev` 侧后续工作

- **STREAM-01 已解锁。** 将 `crates/dg-stream-cheetah/Cargo.toml` 的 cheetah
  rev 从 `182621c` bump 到包含 `cheetah-connector`、RTSP pull、WebRTC push
  及 residual-gap 修复的上游提交；随后在 `dg-stream` 集成：
  - RTSP pull；
  - HTTP-FLV pull；
  - RTMP push；
  - WebRTC push；
  - `LoopbackOptions.preferred_layer =
    LoopbackLayer::EngineOnlyBypassWire` 的 socket-free local loopback test。
- `supports()` 可作为四方向能力判定；`open_pull`/`open_push` 不再需要为
  RTSP/WebRTC 增加临时 unsupported workaround。
- **STREAM-02 仍待处理。** 继续针对 R7 的 duration、origin、side_data、
  精确 flags、`pts_us`/`dts_us` 等字段决定扩展 wire metadata，或固化不保真
  契约。
