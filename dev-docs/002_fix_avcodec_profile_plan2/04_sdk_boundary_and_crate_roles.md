# 04. SDK 边界与 Crate 职责

## 1. `dg-media-avcodec`

该 crate 是 curated facade，不是第二套 codec SDK。职责限定为：

- 转发 Cargo Profile。
- 精确重导出 dyun 需要的高层 SDK 类型。
- 提供 dyun Buffer 与 SDK external descriptor 的所有权适配。
- 隔离 unsafe；公共安全 API 不暴露无约束裸指针。

应重导出的高层类型至少包括 Factory V2、Session Request/Bundle/Report/Error、VideoTranscoder Request/Report、VideoBackendPolicy，以及 dyun bridge 使用的 Packet/Image/config/metadata 类型。避免继续使用不受控的 `pub use avcodec::core::*` 作为唯一接口；先建立精确导出列表和编译测试，再逐步收窄通配导出。

## 2. `dg-media`

负责：

- 稳定 Profile 配置名解析。
- graph element 与有界 pump。
- MediaFrame 元数据生成和校验。
- 调用 facade 的高层 Factory/Transcoder。
- 将 SDK report/error 映射为 dyun diagnostics。

不负责 registry 候选、backend id、fallback、staging 决策和设备 SDK 初始化。

## 3. `dg-cli` 与 `dg-capi`

只公开业务参数：Profile、codec、bitstream、尺寸、像素格式、bitrate、time base。不得新增 backend library path、设备上下文裸句柄或底层初始化参数。确需外部设备句柄时使用版本化 dyun descriptor，并交给 SDK external descriptor。

## 4. 服务对象

在 `dg-media` 增加一个可注入的 `AvcodecSdkService`（名称可遵循仓库惯例），持有：

- 一个由 `default_registry_builder()` 构建的 Registry。
- 已解析的 Profile spec。
- 创建 decoder/encoder/processor bundle 的方法。
- 创建融合 transcoder 的方法。

服务可由 element 共享 Registry，但每个 session 独占其状态。测试通过构造函数注入 mock Registry，不使用全局可变 registry。

## 5. 依赖规则测试

- [ ] `dg-core`、`dg-graph`、`dg-runtime` 不依赖 avcodec。
- [ ] 只有 `dg-media-avcodec` manifest 直接出现外部 `avcodec` 包。
- [ ] `dg-media` 不导入 `avcodec-backend-*`。
- [ ] facade 的公共 API 编译测试覆盖全部高层类型。
- [ ] mock Registry 可替代真实 backend 测试 element。

## 6. 完成条件

- [ ] crate 图满足单向依赖。
- [ ] SDK service 取代重复 registry 初始化和 SessionBuilder。
- [ ] facade 不包含 backend 选择算法。
- [ ] unsafe 仅位于审计过的 external ownership bridge。
