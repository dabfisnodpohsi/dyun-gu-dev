# 02. 当前实现审计与删除清单

## 1. 审计目标

把“dyun 必须保留的应用适配”和“应由 SDK 负责的重复实现”逐符号分类。执行体先完成表格，再删除代码，禁止按文件整体重写导致业务元数据丢失。

## 2. 必须替换的实现

| 位置/符号 | 当前问题 | 目标 |
|---|---|---|
| `dg-media-avcodec/Cargo.toml` | 旧 revision、无条件 jpeg/zune/libyuv、低层 feature | 单一 SDK + Profile 转发 |
| `profile::ProfileDescriptor` | 保存 backend id 数组和单一 memory domain | 上游 Profile V2 |
| `profile_descriptor()` | 重复候选顺序和 fallback | 上游 policy preset |
| `AvcodecSessionBuilder` | 重做 create/trace/failure report | `VideoSessionFactoryV2` |
| `legacy::backend_candidates` | 根据 hw/codec 手工选后端 | Profile 兼容映射 |
| `legacy::create_*` | 循环尝试 registry | SDK Factory |
| `ensure_session_create_supported` | 错误保留已完成上游门禁 | capability/preflight |
| element 初始化 | 每个角色重复构造 registry/config | 共享 SDK service |

## 3. 必须保留的实现

- `dg_core::MediaFrame` 和媒体元数据模型。
- Packet/Image 与 MediaFrame 的字段映射。
- graph element 的输入输出端口、生命周期和调度接口。
- 为 push/poll 差异提供的有界 AsyncPump，但其状态不得解释 backend policy。
- 外部 fd/handle 与 dyun Buffer 所有权桥接。
- CLI/C API 到统一 Profile 配置的解析。

## 4. 源码扫描门禁

建立自动检查，生产源码中以下模式只允许出现在兼容模块或测试 fixture：

```text
"ffmpeg" "x264" "x265" "openh264" "rkmpp" "librga"
"nvcodec" "onevpl" "amf"
create_decoder_with_trace create_encoder_with_trace
BackendSelectionPolicy::Required BackendSelectionPolicy::Ordered
```

允许的例外必须逐行登记原因。Profile 的稳定用户名称可以包含硬件家族名，但不能由 dyun 将其转换成 backend id 数组。

## 5. 审计任务

- [ ] 生成符号级清单：删除、保留、迁移、兼容四类。
- [ ] 为每个删除符号找到上游替代 API 和测试。
- [ ] 标记所有外部公开类型和 feature，确认兼容影响。
- [ ] 标记所有 `unwrap/expect`；生产 element 状态机中改为结构化错误。
- [ ] 统计 bridge 中 Host copy、clone 和 staging 点。
- [ ] 识别测试中依赖旧候选顺序的断言并改为 capability/report 断言。

## 6. 完成条件

- [ ] 删除清单没有包含 MediaFrame/graph 必需职责。
- [ ] 后端选择重复实现均绑定到 SDK 替代入口。
- [ ] 扫描脚本进入 CI。
- [ ] 审计结果覆盖 `INT2-03`。

