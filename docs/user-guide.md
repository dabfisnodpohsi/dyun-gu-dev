# dyun-gu-dev 用户指南

## 1. 环境

- Rust 1.96.1
- 默认/mock 路径无需厂商 SDK
- OpenVINO、RKNN2、TensorRT、Sophon 真实后端需要相应 SDK 与运行设备

```bash
rustup show
cargo build --workspace
```

## 2. 运行示例

仓库提供一个无需模型或硬件的多分支、多算法图：

```bash
cargo run -p dg-cli -- validate --config examples/mock-multi-algorithm.yaml
cargo run -p dg-cli -- run --config examples/mock-multi-algorithm.yaml
```

该图广播三帧张量，同时执行分类后处理和检测、跟踪链路。`mock_inference`
只用于验证编排与算法 element；真实部署使用通用 `inference` element，并为
`dg-cli` 启用对应后端 feature。

## 3. GraphSpec

配置支持 YAML、JSON 和 TOML。最小 YAML：

```yaml
apiVersion: dg/v1
kind: Graph
nodes:
  - name: source
    kind: source
    params:
      count: 1
      shape: [1, 4]
  - name: infer
    kind: mock_inference
    params:
      shape: [1, 4]
      echo_inputs: true
  - name: sink
    kind: sink
    params: {}
connections:
  - source.out -> infer.in
  - infer.out -> sink.in
```

配置加载时会检查未知节点类型、重复节点、端口名称、悬空引用和 DAG 环；注册了
validator 的 element 还会在加载阶段检查参数。内置 `source`、`input`、
`mock_inference`、`sink` 会拒绝未知字段、错误类型和非法枚举值；通用
`inference` 会严格解析对应后端的 `options`，并 preflight 精度、设备和部署模式，
但不会初始化模型或硬件。
`includes`、`variables` 与 `templates` 可用于拆分和复用配置。

顶层 `defaults` 可为支持这些字段的 element（当前为 `inference`）提供
`backend`、`device` 和 `precision` 默认值：

```yaml
defaults:
  backend: mock
  device: cpu
  precision: f32
```

参数优先级为 **node 参数 > template 参数 > 全局 defaults**；defaults 只填充
节点和模板都未提供的字段，不会覆盖已有值。include 文件中的 defaults 低于顶层
配置，顶层文件提供的值优先。`${var}` 变量会在 defaults 注入后替换，因此默认值
也可以引用变量。

配置模型同时接受设计文档中的标准字段别名：`type` 是 `kind` 的别名，
`edges` 是 `connections` 的别名，`vars` 是 `variables` 的别名。旧字段仍然
保持兼容。节点也可以用 `backend`、`device`、`precision` 提供节点级覆盖，
优先级为 **params 显式字段 > 节点级字段 > template 参数 > 全局 defaults**。
`threads` 和 `sink` 字段目前仅保留配置数据，运行时语义属于 CFG-08。
结构化的 `defaults.device`（例如 `{ kind: OpenVino, id: 0 }`）可以解析并
序列化，但调度器级设备含义暂由后续 SCH-* 任务接入；当前只有字符串形式会
注入 `inference` 参数。

执行策略在顶层 `execution` 配置：

```yaml
execution:
  parallel: pipeline
  queue_capacity: 20
```

`pipeline` 使用有界队列提供背压；`task` 在依赖满足后提交到 work-stealing
线程池，并可配置 `workers`；`sequential` 按拓扑顺序在调用线程执行。

```bash
cargo run -p dg-cli -- list-elements
```

导出已注册 element 的参数 JSON Schema，或查看单个 element：

```bash
cargo run -p dg-cli -- schema
cargo run -p dg-cli -- schema --kind media_osd
```

## 4. 后端

| 后端 | crate | 典型目标 | 外部要求 |
|---|---|---|---|
| Mock | `dg-runtime` | CI/开发 | 无 |
| OpenVINO | `dg-openvino` | x86 CPU/GPU | OpenVINO runtime |
| RKNN2 | `dg-rknn` | RK3588 等 SoC | RKNN Toolkit2/runtime |
| TensorRT | `dg-tensorrt` | NVIDIA GPU | CUDA/TensorRT |
| Sophon | `dg-sophon` | BM/SE 设备 | Sophon SDK |

真实后端应在启动时完成设备、精度、部署模式和内存能力校验。缺少 SDK
或设备时返回明确错误，不自动切换到 mock。

单输入模型可直接接入图；多输出会依次发送到同一个 `out` 端口。示例：

```yaml
nodes:
  - name: infer
    kind: inference
    params:
      backend: tensorrt
      model: /opt/models/model.engine
      precision: f16
      device: cuda_gpu
      deploy_mode: host
      reshape: [1, 3, 640, 640]
      options:
        device_id: 0
        workspace_size_mb: 1024
        enable_fp16: true
```

`options` 会按后端严格校验：OpenVINO 支持 `device`；RKNN 支持
`enable_zero_copy`/`dynamic_shape`；TensorRT 支持 `device_id`、
`workspace_size_mb`、`enable_fp16`/`enable_int8`；Sophon 支持 `device_id`。
RKNN/Sophon 的 `core_mask`、通用 `precision`/`device`/`deploy_mode` 位于
`params` 顶层。

```bash
cargo run -p dg-cli --features openvino -- run --config graph.yaml
cargo run -p dg-cli --features rknn -- run --config graph.yaml
cargo run -p dg-cli --features tensorrt -- run --config graph.yaml
cargo run -p dg-cli --features sophon -- run --config graph.yaml
```

真实后端 feature 会链接相应厂商 SDK；默认构建仍保持无 SDK。

## 5. 媒体与流

`dg-media` 注册 `media_decode`、`media_encode`、`media_resize`、`media_osd`。
默认构建可用录制的内存帧验证完整图；`avcodec` feature 提供外部媒体 handle
导入、同内存域共享和跨域 staging fallback。

`dg-stream` 注册 `rtsp_src`、`httpflv_src`、`rtmp_sink`、`webrtc_sink`。
`mock://` URL 使用进程内 `MemoryStreamHub`，适合确定性测试；真实协议 URL 通过
`cheetah` feature 的 connector 接入，未启用时会明确报错。发布前应确认视频/音频
track 已 Ready，并提供 H264/H265/H266/AAC 所需的 codec extradata。

## 6. C API

头文件位于 `crates/dg-capi/include/dg_capi.h`，示例位于
`crates/dg-capi/examples/basic.c`。句柄由对应的 `*_free` 函数释放，失败后可通过
`dg_last_error()` 获取当前线程的错误信息。

`dg_engine_reload_string` / `dg_engine_reload_file` 会对已构建图原位应用新配置，
无需再次 `dg_engine_build`。为避免输入被不同配置解释，仍有待处理输入时 reload
会被拒绝；先执行或排空输入后再更新。`dg-graph::watch` 可监控文件并返回
`GraphDiff`。

## 7. 交叉检查

```bash
rustup target add \
  aarch64-unknown-linux-gnu \
  aarch64-linux-android \
  riscv64gc-unknown-linux-gnu

cargo check --workspace --target x86_64-unknown-linux-gnu
cargo check --workspace --target aarch64-unknown-linux-gnu
cargo check --workspace --target aarch64-linux-android
cargo check --workspace --target riscv64gc-unknown-linux-gnu
```

这些命令验证无 SDK 的默认构建。真实后端交叉链接还需要目标 sysroot、厂商库和
对应 linker。

## 8. 质量与诊断

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

运行时使用 `tracing`。可通过 `RUST_LOG` 和 `-v`/`-vv` 调整日志级别：

```bash
RUST_LOG=dg_graph=debug cargo run -p dg-cli -- -vv run \
  --config examples/mock-multi-algorithm.yaml
```

零拷贝路径应记录内存域、实际传输路径和拷贝次数。若内存域不兼容，框架会显式
走 staging fallback；日志中的“可编译/已规划”不等于已在目标硬件验证。

仓库 CI 覆盖无 SDK 的四目标编译、mock/录制数据测试和供应链检查。RKNN、
TensorRT、Sophon 真实执行、硬件编解码以及端到端零拷贝吞吐仍需对应板卡/GPU
的自托管 runner 验收。
