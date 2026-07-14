# dyun-gu-dev 正确集成 avcodec-rs 高层 SDK 执行计划

## 1. 文档定位

本目录取代 `001_fix_avcodec_profile_plan` 的执行结论，但保留旧目录作为历史记录。旧计划基于 avcodec-rs `8ef5a72`，将 Profile V2、External Packet、RKMPP/RGA 拓扑和 NV device-frame 当作上游门禁；这些能力已经在 avcodec-rs `fc728aa9ea3e0a85401d2cd4de1b762ffcf92a51` 合入，因此不得继续按旧门禁实现。

本计划的唯一目标是：dyun-gu-dev 通过 avcodec-rs 高层 SDK 提供软件和硬件视频解码、编码、图像转换与缩放；外部应用只选择 dyun Profile 和业务参数，不声明、导入、选择或实例化任何底层 codec/backend。

底层库作为 avcodec-rs 的传递依赖是正常现象。验收关注“谁负责选择和操作底层库”，不能以 `Cargo.lock` 是否出现可选 backend 判断封装是否完成。

## 2. 执行规则

1. 严格按 Phase 顺序执行；前一阶段完成条件未满足时不得进入依赖它的阶段。
2. 每个 `[ ]` 是独立评审项；完成后改成 `[x]`，追加 commit、测试命令和结果摘要。
3. 修改前先添加失败测试或依赖树断言，禁止仅凭编译成功宣称集成完成。
4. dyun 仓不得复制 avcodec-rs 的 backend id、候选排序、fallback 算法、Session Factory 或 Adapter Chain。
5. 不带 `fallback` 的 Profile 禁止回退；`allow_staging=false` 禁止隐式 Host copy。
6. 禁止生产路径 `todo!()`、`unimplemented!()`、`unwrap()`、吞错、单次 poll 假设和空 Vec 伪装设备缓冲。
7. 本计划不修改 avcodec-rs；上游缺陷登记为 `UP2-*`，附最小复现并阻塞相应发布项。
8. 不增加容器、demux/mux、RTSP、RTMP、TS、PS 或 `avformat` 能力。
9. 文档和实现不得依赖 vendor、Cargo checkout 或其他不可供执行体访问的本地源码。

## 3. 文档索引

| Phase | 文档 | 交付 |
|---|---|---|
| 0 | [01](01_execution_contract_and_baseline.md)、[02](02_current_implementation_audit.md) | 冻结基线、缺口与删除清单 |
| 1 | [03](03_dependency_and_feature_convergence.md)、[04](04_sdk_boundary_and_crate_roles.md) | 单一依赖和 crate 边界 |
| 2 | [05](05_runtime_profile_contract.md)、[06](06_high_level_session_factory.md) | Profile V2 和 Factory V2 |
| 3 | [07](07_packet_image_and_metadata_bridge.md)、[08](08_decode_encode_elements.md)、[09](09_image_processing_elements.md) | 数据桥接和三类 element |
| 4 | [10](10_transcoder_and_graph_integration.md)、[11](11_hardware_profile_integration.md) | 融合转码和硬件能力 |
| 5 | [12](12_errors_diagnostics_and_observability.md)、[13](13_entrypoints_and_legacy_migration.md) | 诊断、入口和兼容迁移 |
| 6 | [14](14_test_matrix_and_ci.md)、[15](15_release_acceptance_and_rollback.md) | CI、硬件验证和发布 |

## 4. 冻结架构边界

```text
external application
  -> dg-cli / dg-capi / dg-media public Profile
  -> dg-media graph elements + MediaFrame metadata
  -> dg-media-avcodec curated SDK facade + ownership bridge
  -> avcodec high-level SDK (Profile V2 / Factory V2 / Transcoder)
  -> avcodec backend and codec crates
  -> native/vendor runtime
```

外部应用只接触前三层。`dg-media` 可以保留 graph push/poll 适配和 `MediaFrame` 转换，但不得拥有最后三层的后端知识。

## 5. 全局完成定义

- [ ] 01–15 的任务和验收全部完成，无未登记 TODO 或 UP2 阻塞。
- [ ] dyun 固定到包含 avcodec-rs 115 P1–P6 的已验证 immutable revision。
- [ ] `dg-media-avcodec` 的直接 codec 依赖只有 `avcodec` SDK。
- [ ] 新入口只暴露 `avcodec-profile-*`，不要求用户选择 `codec-*`。
- [ ] dyun 源码不存在 backend candidate 数组、自建 SessionBuilder 或 registry 选择循环。
- [ ] decode/resize/encode 全部通过 `VideoSessionFactoryV2` 创建会话。
- [ ] 融合转码通过 `VideoTranscoderRequest` 创建，不重做 Adapter Chain。
- [ ] native-free/software 真实 JPEG、H264、H265 用例通过。
- [ ] RKMPP/RGA、NVCodec、OneVPL、AMF 的声明与 capability、真实硬件结果一致。
- [ ] RK 中间图像链 `copy_count == 0`；NV 只称 device-frame，不称完整 CUDA zero-copy。
- [ ] CLI、C API、配置、示例和 Cargo feature 使用同一组 Profile 名。
- [ ] 默认 workspace 在无 codec/native SDK 环境下可构建和测试。

## 6. 需求覆盖矩阵

| ID | 要求 | 文档 |
|---|---|---|
| INT2-01 | 升级并固定正确 avcodec-rs revision | 01、03、15 |
| INT2-02 | Profile feature 一对一转发 | 03、05、13 |
| INT2-03 | 删除 dyun 后端选择重复实现 | 02、04、06 |
| INT2-04 | 高层 SDK 创建 decode/encode/process | 06、08、09 |
| INT2-05 | Packet/Image 与元数据无损互操作 | 07、08 |
| INT2-06 | 软件与 native-free 完整能力 | 05、08、09、14 |
| INT2-07 | RK/NV/OneVPL/AMF 硬件能力 | 05、11、14 |
| INT2-08 | 诊断、兼容和外部简易入口 | 12、13、15 |

