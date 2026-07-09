/// Deployment mode for inference backends and devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeployMode {
    SoC,
    Host,
}
