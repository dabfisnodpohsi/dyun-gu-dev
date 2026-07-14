# 07. Packet、Image 与媒体元数据桥接

## 1. 边界

桥接是 dyun 必须保留的职责，但只做数据模型和所有权转换，不做 codec、resize、颜色转换或后端选择。

## 2. Packet 映射

MediaFrame→Packet 和反向必须保留：stream index、codec、bitstream format、PTS、DTS、duration、time base、key/lost/corrupt/discontinuity flags、extradata/codec side-data 和 payload 范围。

Host payload 优先使用 `ExternalPacketDescriptor` 或安全拥有型 Buffer；仅当生命周期无法共享且 Profile 允许 copy 时复制。转换结果返回 `TransferReport { copy_count, copied_bytes, source_domain, target_domain, reason }`。

## 3. Image 映射

必须保留：pixel format、coded/display width/height、plane 数量、offset、stride、len、crop、PTS/duration、MemoryDomain、external handle 和 owner guard。

多平面 YUV 不得压平成 packed RGB。padded stride 按有效行宽校验，不要求 `stride == width * bytes_per_pixel`。设备 Image 不得转换为空 Host Vec。

## 4. 所有权

- Host borrowed/owned 路径明确谁持有 backing allocation。
- fd/raw handle 通过 SDK external descriptor 和 drop guard 管理，exactly once 释放。
- clone 只能共享 owner 或明确复制；不得重复关闭 fd。
- submit 返回 Again 时输入所有权仍按 SDK 契约保留。
- flush/reset/drop 释放排队 owner，不泄漏也不提前释放。
- unsafe 块上方写明地址范围、对齐、线程、生命周期和别名不变量。

## 5. Copy/Staging 规则

`allow_staging=false` 时任何跨域 copy 立即返回 topology violation。Host 格式转换是显式 image processor 操作，不得在 bridge 隐藏执行。TransferReport 必须传入 diagnostics 统计。

## 6. 测试

- [ ] Packet 全字段 round-trip。
- [ ] H264/H265 extradata 与 AnnexB/AVCC/HVCC 不混淆。
- [ ] RGB、RGBA、NV12、I420 多平面与 padded stride。
- [ ] external Packet/Image success、Again、error、reset、drop 释放计数。
- [ ] 非法 offset/len/stride、domain/handle 组合拒绝。
- [ ] `allow_staging=false` 的 copy_count 恒为零。
- [ ] 未知 codec/format 返回 Unsupported，不回退 JPEG。

## 7. 完成条件

- [ ] bridge 不创建 processor 或 codec session。
- [ ] 所有数据损失和 copy 都可观测。
- [ ] Host 与设备 handle 路径严格隔离。
- [ ] 无 Rust 引用越过 C ABI。

