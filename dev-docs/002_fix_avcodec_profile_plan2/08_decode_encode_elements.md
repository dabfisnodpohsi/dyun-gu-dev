# 08. Decode 与 Encode Element

## 1. Decode

首个合法 Packet 提供 codec、bitstream、time base 和 extradata；由 SDK service 构建 decoder-only request。Profile 冻结 packet/image domain。输出 Image 转为 MediaFrame 时保留布局和时间信息。

同一 session 后续输入必须保持 stream identity、codec、bitstream 和 time base。变化时返回明确错误或触发上层显式重建，不能隐式新建另一个 backend。

## 2. Encode

首个 Image 提供尺寸、格式、domain 和 timing；配置必须显式或从 frame metadata 解析 bitrate/time base。H264/H265/VPx/AV1 缺少必需 bitrate 时返回 Config，不使用 0。输出 Packet 写回 codec、bitstream、stream index、time base 和 flags。

## 3. Pump 状态机

保留 dyun 有界 AsyncPump，但状态语义与 SDK 一致：

- submit success：输入已消费。
- Again：输入未消费，保存同一对象并先 poll。
- Pending：当前无输出，不是 EOS。
- EndOfStream：仅在 flush 后且排队输入/输出清空时传播。
- flush：停止接收新输入，drain session 至 EOS。
- reset：清空 pending、flush/eos 标志并调用 session reset。

每次 tick 设置最大操作次数，避免 backend 连续 Pending 导致 busy loop。

## 4. CSC 边界

decode element 如配置要求 RGB 输出，必须通过同一 Factory 请求 processor，或将原生 Image 交给独立 resize/convert element。禁止在 bridge 中临时创建 processor。NV device-frame 无 processor 时返回 Unsupported。

## 5. 实施任务

- [ ] `DecoderBackend::ensure_decoder` 改用 SDK service。
- [ ] `EncoderBackend::ensure_encoder` 改用 SDK service。
- [ ] 删除 element 内 registry 构建和 AvcodecSessionBuilder。
- [ ] 消除生产代码中的 session `expect`，用状态错误替代。
- [ ] report 保存到 element 并可查询。
- [ ] 队列容量、tick budget 和错误重试均有上限。

## 6. 测试

- JPEG、H264、H265 decode/encode 真实 fixture。
- mock Again→Ready、Pending→Ready、flush 多输出、reset 后重用。
- 码流参数变化、time base 冲突、缺 bitrate、损坏 packet。
- backend device lost、queue full、out of memory。
- 连续 1000 帧无丢失、重复和乱序。

## 7. 完成条件

- [ ] element 只封装 graph 生命周期，不封装后端选择。
- [ ] 输入所有权和 EOS 语义通过状态机测试。
- [ ] report、实际 backend 和 Profile 一致。

