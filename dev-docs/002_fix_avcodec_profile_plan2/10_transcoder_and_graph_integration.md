# 10. Transcoder 与 Graph 集成

## 1. 两种模式

独立 `media_decode → media_resize → media_encode` 继续作为 graph 的标准组合，各节点使用 Factory V2。只有业务明确请求压缩流直接转码且无需暴露中间 frame 时，新增或改造融合入口使用 `VideoTranscoderRequest`。

不得把所有独立 element 强制合并，也不得在 dyun 重写 SDK `VideoFrameAdapter`。

## 2. 融合 Transcoder 请求

请求包含同一个 `VideoProfileDescriptor`、decoder config、encoder config 和 `VideoTranscodeOptions`。options 只表达目标尺寸、格式和是否允许 linked mode；backend policy 和 staging 来自 Profile。

构建后保存 `VideoTranscoderBuildReport`：Passthrough、Linked 或 Adapted 模式、adapter chain、allow_staging。调用方不根据 backend 名改变业务逻辑。

## 3. Graph 行为

- 输入/输出仍为 MediaFrame encoded payload。
- Packet bridge 在 graph 边界执行一次。
- 背压、Again、flush 和 EOS 交给 SDK transcoder，再映射到 graph poll。
- adapter chain 的 Image 不往返 MediaFrame，避免无意义 copy。
- graph stop/reset 时 drain 或 reset transcoder，不能遗留 session owner。

## 4. 选择规则

- 单纯 decode、encode、resize 使用 Factory，不使用 Transcoder。
- codec passthrough 只有 SDK report 为 Passthrough 时成立。
- linked backend 由 SDK capability 决定。
- `allow_staging=false` 且无法形成适配链时构建失败。
- NV device-frame 请求 resize 时，在无 CUDA processor 环境稳定失败。

## 5. 测试

- [ ] passthrough、linked、adapted 三种 mock report。
- [ ] decode→CSC→resize→encode adapter chain。
- [ ] Again 交错、flush drain、多 packet 输出。
- [ ] graph cancel/reset/drop 无泄漏。
- [ ] 禁止 staging 的负向测试。
- [ ] 上游偶发 diagnostics 用例重复执行并登记结果。

## 6. 完成条件

- [ ] dyun 没有自建完整转码状态机。
- [ ] 融合链 report 可从 diagnostics 查询。
- [ ] 独立 element 与融合入口的媒体语义一致。
