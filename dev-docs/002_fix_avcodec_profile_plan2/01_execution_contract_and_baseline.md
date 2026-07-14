# 01. 执行契约与技术基线

## 1. 已确认事实

- dyun 当前 HEAD：`e09eb2af0971ae2d86a6efc8ef62e2092324951c`。
- dyun 当前固定 avcodec-rs：`8ef5a72a50b396bad1a670fc2757893c059191a4`。
- avcodec-rs 集成候选：`fc728aa9ea3e0a85401d2cd4de1b762ffcf92a51`。
- 旧 revision 不包含 `VideoSessionFactoryV2`、`VideoIoMemoryPlan`、`ExternalPacketDescriptor` 和 `profile-nvcodec-device-frame`。
- 当前 avcodec-rs 已公开上述 API，但完整 `profile-native-free` 测试曾出现一个顺序相关/偶发失败；单用例复跑通过。此风险登记为 `UP2-TEST-01`。
- dyun 工具链固定 Rust `1.94.1`，执行命令使用 `RUSTUP_TOOLCHAIN=stable` 验证实际版本，避免 rustup 镜像误报不存在。

## 2. 基线冻结任务

- [ ] 记录执行前 dyun HEAD、工作区状态和所有未提交用户改动；不得覆盖无关改动。
- [ ] 记录 `rustc --version`、`cargo --version`、目标三元组和操作系统。
- [ ] 保存旧 revision 下 native-free、software 的 `cargo tree -e features` 输出。
- [ ] 证明当前 `dg-media-avcodec` 只有一个直接 codec 依赖，但 feature 映射仍错误。
- [ ] 记录当前 Stub：`RkmppZeroCopy`、`NvcodecDeviceFrame` 创建必定失败。
- [ ] 记录 `profile.rs`、`session.rs`、`legacy.rs` 和 `avcodec.rs` 中重复实现符号。

## 3. 上游接纳门禁

将 `fc728aa...` 作为迁移起点。更新依赖前必须验证：

1. 目标 revision 可通过 Git 获取且 `Cargo.lock` 精确锁定同一 hash。
2. `cargo check -p avcodec --no-default-features --features profile-native-free` 可通过。
3. Profile V2、Factory V2、External Packet 和 device-frame 符号可从 `avcodec` 顶层或 `avcodec::core` 访问。
4. focused consumer 测试通过；不要求 dyun 修复上游测试。
5. `UP2-TEST-01` 若可重复，提交包含命令、seed/顺序和日志的上游缺陷；仅阻塞使用受影响 Transcoder 诊断的发布项。

## 4. 禁止项

- 不通过复制上游源码绕过 revision 升级。
- 不在 dyun 引入 `avcodec-backend-*` 或 `avcodec-codec-*`。
- 不把 Cargo.lock 中的可选依赖数量当作实际编译依赖数量。
- 不为通过测试虚报硬件 capability 或自动回退软件。
- 不修改 graph/stream 的协议和容器职责。

## 5. 完成条件

- [ ] 基线报告可复现且包含命令、输出摘要和 commit。
- [ ] `INT2-01` 的目标 revision 与已知风险明确。
- [ ] 所有后续任务使用同一工具链和依赖基线。
- [ ] 用户已有工作区变更未被修改。

