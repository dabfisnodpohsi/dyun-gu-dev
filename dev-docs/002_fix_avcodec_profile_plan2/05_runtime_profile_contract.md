# 05. 运行时 Profile 契约

## 1. 稳定 Profile 名

运行配置只接受：`native-free`、`software`、`rkmpp-host`、`rkmpp-host-fallback`、`rkmpp-zero-copy`、`nvcodec-host`、`nvcodec-host-fallback`、`nvcodec-device-frame`、`onevpl-host`、`onevpl-host-fallback`、`amf-host`、`amf-host-fallback`。

解析大小写可不敏感，但 diagnostics 始终输出规范小写名。Profile 未编译时返回 Config 错误并列出已编译 Profile；编译多个且配置未指定时返回歧义错误。

## 2. 薄 Profile Spec

`AvcodecProfile` 只负责稳定名称和 feature 可用性。其解析结果转换为 SDK `VideoProfileDescriptor`，策略必须来自 `VideoBackendPolicy` 构造器：

- software/native-free：`VideoBackendPolicy::software()`。
- RK Host：`rkmpp_host(false/true)`。
- RK zero-copy：`rkmpp_zero_copy()`。
- NV Host：`nvcodec_host(false/true)`。
- NV device-frame：`nvcodec_device_frame()`。
- OneVPL：`onevpl_host(false/true)`。
- AMF：`amf_host(false/true)`；非 fallback 不宣称 AMF decode。

dyun 禁止解构 policy 后重排或追加 backend id。

## 3. 内存拓扑

| Profile | Decoder Packet | Decoder Image | Processor | Encoder Image | Encoder Packet | staging |
|---|---|---|---|---|---|---|
| native-free/software | Host | Host | Host→Host（按需） | Host | Host | false |
| RK Host | Host | Host | Host→Host | Host | Host | SDK 能力决定，配置显式 |
| RK zero-copy | Host | DrmPrime | DrmPrime→DmaBuf | DmaBuf | Host | false |
| NV Host | Host | Host | Host→Host | Host | Host | SDK 能力决定，配置显式 |
| NV device-frame | Host | CudaDevice | 无 | CudaDevice | Host | false |
| OneVPL Host | Host | Host | Host→Host | Host | Host | SDK 能力决定，配置显式 |
| AMF Host | Host | Host | Host→Host | Host | Host | SDK 能力决定，配置显式 |

Host Profile 的 staging 默认值必须由集成测试验证后冻结；不得为让创建成功而无条件设 true。跨域 copy 只有 Profile 明确允许且 report 可见时才合法。

## 4. 操作支持

- decode-only：只请求 decoder，不伪造 encoder config。
- encode-only：只请求 encoder。
- resize/CSC：只请求 processor，并提供 `ImageOpKind`、输入/输出域和格式。
- NV device-frame 无 CUDA processor，resize/CSC 返回 Unsupported；不得自动下载到 Host。
- AMF 若 SDK capability 不支持 decode，decode 返回 Unsupported；fallback 仅由 fallback Profile 生效。

## 5. 配置冲突

`profile` 与旧 `hw` 同时出现时返回 Config 错误。新配置不允许 `backend` 字段。`allow_staging` 不能覆盖严格 zero-copy Profile；用户请求与 Profile 不一致时在 session 创建前失败。

## 6. 任务与验收

- [ ] 删除 backend hint 数组和旧单域 `ProfileDescriptor`。
- [ ] 实现 Profile→SDK descriptor 的唯一转换函数。
- [ ] 为十二个 Profile 添加名称、feature 和 topology 表驱动测试。
- [ ] 验证非 fallback/fallback 的 policy report。
- [ ] 验证 NV processor、AMF decode 等 Unsupported 边界。
- [ ] Profile descriptor 调用上游 `validate()`。

