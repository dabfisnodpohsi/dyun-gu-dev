# 11. 硬件 Profile 集成

## 1. 通用规则

硬件 Profile 是否可用由编译 feature、SDK capability、驱动/runtime 和设备权限共同决定。缺少任一条件返回结构化 Unsupported/NotAvailable；非 fallback 不转软件。普通 CI 可解释跳过，专用硬件 runner 缺失预期设备必须失败。

## 2. RKMPP/RGA

Host Profile 验证 decode、encode、Resize/CSC。zero-copy 固定拓扑：

```text
Host compressed Packet
 -> RKMPP decoder
 -> DrmPrime NV12 Image
 -> librga Resize/CSC
 -> DmaBuf NV12 Image
 -> RKMPP encoder
 -> Host compressed Packet
```

必须检查 fd owner、plane offset/stride、格式、尺寸对齐和同步契约。中间 Image 不允许 Host map/memcpy；TransferReport 和 SDK diagnostics 的 Host copy/staging bytes 必须为零。

## 3. NVCodec

Host Profile 允许 Host Packet/Image，并可使用 libyuv 处理。device-frame 固定为 Host Packet→CudaDevice Image→Host Packet；它不是完整 CUDA Packet zero-copy。

Cuda Image 必须保留 device/context identity、plane/pitch 和 owner。Host session 拒绝 Cuda Image，device session 拒绝 Host plane。无 CUDA processor 时 Resize/CSC 返回 Unsupported，禁止隐式 download/upload。

## 4. OneVPL

只宣称 SDK capability 实际支持的 Host decode/encode；图片处理使用 Host libyuv。检查驱动、设备枚举和 session 创建。fallback Profile 可选 FFmpeg，非 fallback 只能报告 OneVPL 或失败。

## 5. AMF

按上游真实 capability 暴露，优先验证 encode。`VideoBackendPolicy::amf_host(false)` 当前 decoder 策略不能作为“AMF 解码保证”；dyun 文档、CLI 和 capability 输出不得宣称未验证 decode。需要 decode 时使用明确支持的 capability 或 fallback Profile。

## 6. 测试环境合同

每类 runner 记录：OS/arch、设备型号、驱动/runtime 版本、权限、feature、测试 fixture hash、SDK revision。skip 原因枚举为 FeatureDisabled、DeviceMissing、DriverMissing、PermissionDenied、RuntimeMissing；禁止自由文本掩盖失败。

## 7. 验收任务

- [ ] RK Host decode/resize/encode smoke。
- [ ] RK zero-copy copy_count/staging bytes 为零。
- [ ] RK fd 泄漏和 8 小时 reset/flush soak。
- [ ] NV Host 和 device-frame 分别测试。
- [ ] NV context/domain/pitch 错误拒绝。
- [ ] OneVPL 非 fallback/fallback 选择报告。
- [ ] AMF encode 与不支持 decode 的诚实报告。
- [ ] 所有硬件 Profile 无设备时错误可解释。

## 8. 完成条件

- [ ] 实际 backend、MemoryDomain 和 Profile report 一致。
- [ ] 无能力虚报或静默软件回退。
- [ ] 零拷贝声明有可复现测量证据。

