# 03. 依赖与 Profile Feature 收敛

## 1. 目标依赖

`dg-media-avcodec` 是唯一直接依赖 avcodec-rs 的 crate：

```toml
avcodec = { git = "https://github.com/TimothyWalker6922/avcodec-rs-develop.git", rev = "fc728aa9ea3e0a85401d2cd4de1b762ffcf92a51", package = "avcodec", default-features = false, optional = true }
```

不得在该依赖项上无条件启用 `jpeg`、`zune`、`libyuv`。最终 revision 如因已确认的上游修复前移，必须替换为不可变 hash，并在本文件执行记录中说明原因和验证结果。

## 2. Feature 映射

每个 dyun feature 只转发同名上游 Profile：

```text
native-free              -> avcodec/profile-native-free
software                 -> avcodec/profile-software
rkmpp-host               -> avcodec/profile-rkmpp-host
rkmpp-host-fallback      -> avcodec/profile-rkmpp-host-fallback
rkmpp-zero-copy          -> avcodec/profile-rkmpp-zero-copy
nvcodec-host             -> avcodec/profile-nvcodec-host
nvcodec-host-fallback    -> avcodec/profile-nvcodec-host-fallback
nvcodec-device-frame     -> avcodec/profile-nvcodec-device-frame
onevpl-host              -> avcodec/profile-onevpl-host
onevpl-host-fallback     -> avcodec/profile-onevpl-host-fallback
amf-host                 -> avcodec/profile-amf-host
amf-host-fallback        -> avcodec/profile-amf-host-fallback
```

同样的 feature 名必须逐层转发到 `dg-media`、`dg-cli`、`dg-capi`，不得在中间层展开成低层 feature。

## 3. 兼容 Feature

旧 `avcodec` 映射到 `avcodec-profile-native-free`，保留一个发布周期。旧 `codec-*` 保留一个发布周期但不得出现在默认、示例或 CI 主路径；发布说明必须声明它们只用于迁移且下一兼容版本删除。

兼容 feature 不得让业务源码恢复手工后端选择。即便启用旧名，也必须经过新 Factory/配置路径。

## 4. 依赖树合同

添加脚本或集成测试执行 `cargo tree` 并断言：

- native-free 不激活 ffmpeg、x264、x265、openh264、rkmpp、nvcodec。
- software 使用上游 `profile-software`，不再使用 `software-default`。
- 单一硬件 Profile 不激活其他硬件家族。
- fallback Profile 只比非 fallback 增加其公开的软件回退栈。
- 无 Profile 的默认 workspace 不激活 avcodec。

Cargo.lock 可包含可选包，断言必须基于启用后的 feature tree 或实际编译单元。

## 5. 实施任务

- [ ] 更新 revision 并重生成 Cargo.lock。
- [ ] 删除依赖项上的基础 backend feature。
- [ ] 在四层 Cargo.toml 一对一转发 Profile。
- [ ] 删除新文档和示例中的低层 feature 用法。
- [ ] 添加 feature 单开和冲突检查。
- [ ] 多个 Profile 同时编译时要求运行配置显式选择，不静默取第一个。

## 6. 完成条件

- [ ] 外部应用只需启用一个 dyun `avcodec-profile-*`。
- [ ] 依赖树合同自动化通过。
- [ ] 默认 workspace 无 native codec/SDK 要求。
- [ ] Cargo.lock 和 manifest 使用同一 immutable revision。

