# 09. 图片处理 Element

## 1. 支持操作

第一阶段覆盖 Resize 和 CSC；旋转、crop、OSD 只有在 SDK capability 与真实实现均存在时开放。输入请求明确 source Image、operation、目标格式/尺寸和期望输出 domain。

## 2. Profile 路径

- native-free/software：Host→Host，选择 SDK 注册的 libyuv 或 Profile 允许的处理器。
- RK Host：Host→Host，由 SDK 选择 librga 或 fallback processor。
- RK zero-copy：DrmPrime→DmaBuf，必须是 SDK capability 声明的真实转换。
- NV Host：Host→Host libyuv。
- NV device-frame：无 CUDA processor，Resize/CSC 返回 Unsupported。
- OneVPL/AMF Host：Host→Host libyuv；不得声称硬件图像处理。

## 3. 创建与执行

ResizeCore 创建 processor-only `VideoSessionRequest`，设置 `ImageProcessorConfig.target_op` 和 Profile 的 processor I/O domain。删除直接 `build_image_processor` 的 dyun builder 调用。

输出尺寸、格式、plane 和 stride 以 SDK Image 为准，并验证与请求一致；后端可合法增加对齐，但不得静默改变 visible size/crop。

## 4. 错误与约束

- width/height 为零、溢出或不满足后端对齐时返回 InvalidArgument。
- 不支持的格式/操作/domain transition 返回 Unsupported，并包含 capability trace。
- `allow_staging=false` 时不安装 stage-to-host hook。
- fallback 只在 Profile 名允许时发生，report 记录实际 processor。

## 5. 测试

- [ ] RGB/RGBA/NV12/I420 resize 与 CSC。
- [ ] 奇数尺寸、padded stride、多平面和 crop。
- [ ] processor Again/Pending/flush/reset。
- [ ] RK DrmPrime→DmaBuf mock 生命周期和真实硬件 copy 计数。
- [ ] NV device-frame Unsupported，且无 Host allocation。
- [ ] 非 fallback 不选择 libyuv 替代硬件 processor。

## 6. 完成条件

- [ ] 所有 processor 由 Factory V2 创建。
- [ ] bridge 不隐藏图片处理。
- [ ] capability、report 和实际 output domain 一致。

