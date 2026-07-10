# dyun-gu-dev

Rust 多芯片推理框架（OpenVINO / TensorRT / RKNN2 / Sophon）。

## 快速开始

```bash
cargo run -p dg-cli -- validate --config examples/mock-multi-algorithm.yaml
cargo run -p dg-cli -- run --config examples/mock-multi-algorithm.yaml
cargo run -p dg-cli -- list-elements
```

默认构建不依赖厂商 SDK，并使用 mock 后端验证图执行、算法后处理与多分支编排。
真实后端通过各 crate 的 feature 和对应 SDK 环境启用。

- [用户指南](docs/user-guide.md)
- [设计方案与里程碑](docs/design.md)
- [C API 示例](crates/dg-capi/examples/basic.c)

## 质量门禁

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
