# 13. 外部入口与旧接口迁移

## 1. 统一用户接口

CLI、C API、配置文件和 Rust API 使用同一 Profile 名集合。用户只配置 Profile、codec、bitstream、尺寸/格式、bitrate、time base 和 element 队列；不配置 backend id 或底层库路径。

示例：

```text
dyun --features avcodec-profile-software ... profile=software
dyun --features avcodec-profile-rkmpp-zero-copy ... profile=rkmpp-zero-copy
```

实际命令语法按现有 CLI 保持，但概念不得偏离。

## 2. 配置优先级

只有新 `profile` 生效。旧 `hw` 或 backend 参数单独出现时通过兼容 mapper 转成 Profile 并产生一次弃用告警；与 `profile` 同时出现返回冲突错误。环境变量、CLI 和配置文件遵循项目现有优先级，不另建隐式 fallback。

## 3. 兼容周期

- `avcodec` feature → native-free Profile。
- 旧 `codec-*` feature 保留一个发布周期，仅保证能够编译并进入新 SDK 路径，不保证旧 backend 强制语义继续存在。
- `hw=auto/cpu/rkmpp/nvcodec/...` 映射到书面定义的 Profile；无法无歧义映射时返回迁移错误。
- 下一兼容版本删除旧 feature、旧 hw mapper、legacy.rs 和相关测试。

弃用文档必须列出替代名称、首次弃用版本、计划移除版本和行为差异。

## 4. C API

C 配置结构保持 size/version 防御，新增 Profile 字符串或稳定 enum 时使用新版本字段；旧调用按兼容 mapper 处理。空指针、未知版本、未终止字符串和未知 Profile 必须拒绝。C API 不暴露 Rust 引用或 backend session。

## 5. 示例

新增可编译示例：native-free decode/resize/encode、software H264 transcode、RK Host、RK zero-copy、NV Host、NV device-frame decode-only、OneVPL Host、AMF encode、preflight failure report。硬件示例无设备时输出规范 skip。

## 6. 任务与验收

- [ ] 四类入口共享一个 Profile parser。
- [ ] feature 名与运行 Profile 名矩阵测试。
- [ ] profile/hw 冲突测试。
- [ ] legacy consumer 编译和弃用告警测试。
- [ ] 示例不导入任何底层 backend crate。
- [ ] 用户指南不要求安装未选择 Profile 的 SDK。

## 7. 完成条件

- [ ] 新应用只理解 Profile，不理解底层 codec 库。
- [ ] 兼容入口也经过 Factory V2。
- [ ] 移除时间和回滚方式明确。

