# 06. 高层 Session Factory 集成

## 1. 目标

所有 decoder、encoder、image processor 会话由 `VideoSessionFactoryV2` 创建。删除 `AvcodecSessionBuilder` 及所有手工 `create_*_with_trace` 循环。

## 2. SDK Service 接口

建议在 `dg-media` 内部定义：

```rust
pub(crate) struct AvcodecSdkService {
    registry: dg_media_avcodec::Registry,
}

impl AvcodecSdkService {
    pub fn build(&self, request: VideoSessionRequest)
        -> Result<VideoSessionBundle>;
}
```

真实构造器使用 facade 的 `default_registry_builder()`；测试构造器接收 Registry。错误转换保留 `VideoSessionBuildError.role`、failure report 和 context。

## 3. 请求构建

调用顺序固定：

1. 解析并检查编译期 Profile。
2. 创建 `VideoProfileDescriptor` 和 `VideoIoMemoryPlan`。
3. 从 stream/frame 元数据构造角色 config。
4. Packet 输入域、Image 域、Packet 输出域和 `allow_staging` 与 Profile 对齐。
5. 只设置实际请求的 role；不需要 processor 时保持 `None`。
6. 调用一次 Factory `build()`。
7. 保存 bundle report，再将对应 session 移入 element backend。

## 4. 生命周期与失败原子性

- element 初始化失败不保留半创建 session。
- lazy decode/encode 可以等待首个输入确定 codec/尺寸，但 Profile 和 Registry 必须在 element 创建时冻结。
- 同一 stream 的 codec、bitstream、time base 或图像布局发生不兼容变化时返回重配置需求，不静默复用旧 session。
- reset 只重置 session/element 状态，不重新选择另一个 backend。
- device lost 后是否重建由上层显式策略决定，非 fallback Profile 不自动切软件。

## 5. 删除任务

- [ ] 删除 `SessionRole` 和 dyun 自建 `SessionBuildReport`，改用 SDK report 或无损包装。
- [ ] 删除 `select_with_hints`、`apply_profile_*` 和 `map_selection_failure` 的重复部分。
- [ ] 删除 `legacy.rs` 的 registry 遍历。
- [ ] 将 decode、encode、resize、CSC 初始化统一接入 SDK service。
- [ ] 测试中用 mock backend capability 驱动选择，不断言硬编码顺序。

## 6. 测试

- decode-only、encode-only、processor-only、三角色 bundle。
- Profile/config domain 不一致在 submit 前失败。
- 多候选失败保留 SDK SelectionTrace。
- 创建中途失败不泄漏已创建 mock session。
- 两个 element 共享 Registry 但 session 状态隔离。
- 非 fallback Profile 永不出现其他 backend report。

## 7. 完成条件

- [ ] 生产代码只有 Factory V2 创建会话。
- [ ] dyun 不构造 `BackendSelectionPolicy::Required/Ordered`。
- [ ] report 与 element 实际 session 一致。
- [ ] 所有失败保留角色、Profile、codec、domain 和 operation。

