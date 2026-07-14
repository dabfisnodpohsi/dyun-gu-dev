# 12. 错误、诊断与可观测性

## 1. 错误映射

SDK 错误映射到 dyun 时保留：error kind/detail、operation、role、Profile、backend、codec、source/target format、MemoryDomain、allow_staging 和 SelectionFailureReport。不得只保留 `to_string()`。

稳定错误类别：Config、InvalidArgument、Unsupported、Again、EndOfStream、Backend、DeviceLost、OutOfMemory、Timeout、TopologyViolation。Again/EOS 在 pump 内作为控制流，只有非法状态才对外报错。

## 2. Build Report

每个 element 保存 SDK report，至少输出：Profile、选中角色/backend、六方向 I/O domain、processor 操作、staging 计划、legacy warning 和候选淘汰摘要。不得输出裸指针、fd 值、设备句柄或 payload。

融合转码额外输出 mode 和 adapter chain。diagnostics API 使用拥有型快照，不能返回 session 内部引用。

## 3. 指标

- session create success/failure，标签限制为稳定枚举。
- submit Again、poll Pending、queue high-water、flush latency。
- Host copy count/bytes、staging count/bytes、external import count。
- frames/packets in/out/drop/corrupt。
- device lost、fallback occurrence 和 reset。

禁止以 stream id、文件名或错误字符串作为无界标签。

## 4. 日志

session 创建记录一次 INFO；候选失败摘要为 DEBUG；设备丢失、非法 fallback、copy policy 违约为 WARN/ERROR。逐帧日志默认关闭。日志必须包含 graph/element/stream correlation id，但不包含媒体数据。

## 5. 任务与测试

- [ ] 替换字符串化 `map_selection_failure`。
- [ ] SDK report 无损包装并接入 CLI diagnostics。
- [ ] copy/staging 计数由 bridge 和 SDK report 共同校验。
- [ ] 错误 serde/C ABI 映射稳定测试。
- [ ] 敏感句柄和 payload 不出现在 Debug/日志快照。
- [ ] 指标 cardinality 测试。

## 6. 完成条件

- [ ] 任一创建/运行失败可定位到角色、Profile 和约束。
- [ ] 零拷贝与 fallback 可被指标证明。
- [ ] diagnostics 不改变 session 生命周期。

