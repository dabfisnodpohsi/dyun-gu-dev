# 14. 测试矩阵与 CI

## 1. 分层

### 单元测试

Profile 解析/feature、metadata mapper、checked arithmetic、AsyncPump、错误映射和 owner exactly-once。

### 契约测试

mock backend 运行 Factory V2 的 decode-only、encode-only、processor-only 和 bundle；验证 capability、report、fallback 和 domain。

### 软件真实媒体

JPEG、H264、H265 的 decode、encode、round-trip；Host Resize/CSC；损坏、截断和参数变化 fixture。fixture 记录许可证、hash 和生成方式。

### 硬件测试

按 11 文档在专用 runner 执行，普通 CI 只运行探测与确定性 skip 测试。

## 2. Feature 矩阵

每个 Profile 单独 `cargo check` dg-media、dg-cli、dg-capi。额外验证：无 Profile、legacy alias、一个 fallback、两个 Profile 同时编译且运行时显式选择。禁止 `--all-features` 代替单 Profile 测试。

## 3. Dependency Contract

CI 解析 `cargo tree -e features`：

- dyun 直接 codec 包仅 avcodec。
- native-free 无 native runtime backend。
- software 不展开 `software-default`。
- RK/NV/OneVPL/AMF 单开互不污染。
- 默认构建不运行厂商 build script。

## 4. 状态机场景

表驱动覆盖 submit success/Again/error 与 poll Ready/Pending/EOS 的交叉组合；flush 前后多输出；reset；queue full；device lost。断言输入消费次数、输出顺序、drop 计数和 tick 上限。

## 5. 稳定性与性能

- 软件 10k 帧无泄漏/乱序。
- 硬件 8 小时 soak，fd/device buffer 不增长。
- copy/staging bytes 基线。
- 1080p decode→resize→encode throughput/latency 只在稳定 runner 设阈值。
- 上游已知 Transcoder diagnostics 用例重复至少 20 次；若不稳定登记 UP2-TEST-01。

## 6. CI Jobs

- format-check、clippy、default workspace test。
- profile-feature-matrix。
- dependency-contract。
- native-free-real-media、software-real-media。
- legacy-consumer。
- external-owner/Miri（环境支持时）。
- RK、NV、OneVPL、AMF 专用 runner。

## 7. 完成条件

- [ ] INT2-01～INT2-08 均绑定自动化测试。
- [ ] 普通 CI 无硬件可通过且 skip 可解释。
- [ ] 专用 runner 不允许全部 skip 后成功。
- [ ] 零拷贝和 fallback 有正反向证据。
- [ ] 无 flaky 测试被简单重试掩盖。

