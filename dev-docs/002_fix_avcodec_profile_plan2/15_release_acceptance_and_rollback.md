# 15. 发布验收、升级与回滚

## 1. 发布前顺序

1. 完成 01–14 所有清单并登记 commit。
2. 确认 Cargo.toml/Cargo.lock 固定同一 avcodec-rs hash。
3. 生成 feature/dependency/API diff。
4. 运行默认、软件、所有硬件可用矩阵。
5. 检查生产源码无 backend candidate、旧 SessionBuilder 和 Stub gate。
6. 更新 README、用户指南、配置 schema、C header 和 changelog。
7. 使用一个不依赖 avcodec 的外部 consumer 构建并运行三个软件 element。

## 2. 发布阻断项

- SDK revision 无法复现或使用浮动 branch/tag。
- 非 fallback 选择软件后端。
- capability 与实际 session 不一致。
- `allow_staging=false` 出现 Host copy。
- RK/NV 声明与真实 domain 不一致。
- External owner 泄漏、double free 或 use-after-free。
- Again/EOS 导致丢帧、重复或死循环。
- 新入口仍要求用户启用多个 `codec-*`。
- 未登记的 UP2 缺陷影响目标 Profile。

## 3. 回滚

依赖升级和业务迁移分成可回滚提交：依赖/feature、facade、Profile/Factory、bridge/element、入口/兼容、测试/文档。回滚不得恢复旧 revision 与新代码的混合状态。

若某硬件 Profile 未通过，只关闭该 Profile 的发布声明和入口，不回退整个 SDK，也不虚报 capability。软件 Profile 必须保持可独立发布。

## 4. 兼容移除

首个新版本保留旧 `avcodec`、`codec-*` 和 `hw` mapper并告警。下一个约定兼容版本在使用遥测/consumer 审计确认无调用后删除。删除作为独立 breaking change，不与新的 backend 功能混合。

## 5. 最终验收记录模板

```text
dyun commit:
avcodec-rs commit:
rust/cargo/target:
enabled profile:
selected roles/backends:
I/O memory topology:
copy/staging result:
test commands and result:
hardware/driver/runtime:
known limitations/UP2 issues:
reviewer/date:
```

## 6. 最终完成清单

- [ ] 外部 consumer 只有 dyun 依赖和一个 Profile feature。
- [ ] dyun 不包含底层 codec 选择知识。
- [ ] decode、encode、image process 均由高层 SDK 创建和执行。
- [ ] 软件真实媒体与目标硬件矩阵通过。
- [ ] diagnostics 能证明实际 backend、domain、copy 和 fallback。
- [ ] 文档、示例、CLI/C API 与实现一致。
- [ ] 回滚演练成功且不破坏默认无 codec 构建。
