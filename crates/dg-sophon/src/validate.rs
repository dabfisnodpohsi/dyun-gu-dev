//! Runtime capability and precision validation for the Sophon backend.
//!
//! The checks are pure (no FFI) so they run in the no-hardware CI build. They
//! reject unsupported precision / device / deployment combinations up front
//! with errors that name the offending field, instead of failing deep inside
//! the vendor SDK.

use dg_core::DeployMode;
use dg_runtime::{
    supports_deployment, supports_device, supports_precision, BackendKind, Error, Result,
    RuntimeOption, SophonOptions,
};

use crate::convert::SophonDataType;

/// Validates a [`RuntimeOption`] plus the resolved [`SophonOptions`] against the
/// backend's static capability record.
pub fn validate_options(option: &RuntimeOption, sophon: &SophonOptions) -> Result<()> {
    if let Some(precision) = option.precision {
        if !supports_precision(BackendKind::Sophon, precision) {
            return Err(Error::UnsupportedPrecision(precision));
        }
        // The precision must also be expressible as a BMRuntime element type.
        SophonDataType::from_data_type(precision)?;
    }
    if let Some(device) = option.device {
        if !supports_device(BackendKind::Sophon, device) {
            return Err(Error::UnsupportedDevice(device));
        }
    }
    if let Some(deploy_mode) = option.deploy_mode {
        if !supports_deployment(BackendKind::Sophon, deploy_mode) {
            return Err(Error::UnsupportedDeployment(deploy_mode));
        }
        if deploy_mode != sophon.deploy_mode {
            return Err(Error::InvalidOption(format!(
                "RuntimeOption deploy mode {deploy_mode:?} conflicts with Sophon option {:?}",
                sophon.deploy_mode
            )));
        }
    }
    if !supports_deployment(BackendKind::Sophon, sophon.deploy_mode) {
        return Err(Error::UnsupportedDeployment(sophon.deploy_mode));
    }
    Ok(())
}

/// Ensures the requested deployment mode matches the mode this crate was
/// compiled for (SoC vs Host link differs in the vendor SDK).
pub fn validate_deploy_mode(requested: DeployMode, compiled: DeployMode) -> Result<()> {
    if requested == compiled {
        Ok(())
    } else {
        Err(Error::UnsupportedDeployment(requested))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dg_core::{DataType, DeviceKind};
    use dg_runtime::{BackendOptions, ModelSource, RuntimeOption};

    fn base_option() -> RuntimeOption {
        RuntimeOption::new(
            BackendKind::Sophon,
            ModelSource::Bytes(vec![0u8; 4]),
            BackendOptions::Sophon(SophonOptions::default()),
        )
    }

    #[test]
    fn accepts_supported_configuration() {
        let sophon = SophonOptions::default();
        let option = base_option()
            .with_precision(DataType::F16)
            .with_device(DeviceKind::SophonTpu)
            .with_deploy_mode(DeployMode::Host);
        validate_options(&option, &sophon).unwrap();
    }

    #[test]
    fn rejects_unsupported_precision() {
        let sophon = SophonOptions::default();
        let option = base_option().with_precision(DataType::I4);
        assert!(matches!(
            validate_options(&option, &sophon),
            Err(Error::UnsupportedPrecision(_))
        ));
    }

    #[test]
    fn rejects_unsupported_device() {
        let sophon = SophonOptions::default();
        let option = base_option().with_device(DeviceKind::CudaGpu);
        assert!(matches!(
            validate_options(&option, &sophon),
            Err(Error::UnsupportedDevice(DeviceKind::CudaGpu))
        ));
    }

    #[test]
    fn rejects_conflicting_deploy_mode() {
        let sophon = SophonOptions {
            deploy_mode: DeployMode::Host,
            ..SophonOptions::default()
        };
        let option = base_option().with_deploy_mode(DeployMode::SoC);
        assert!(matches!(
            validate_options(&option, &sophon),
            Err(Error::InvalidOption(_))
        ));
    }

    #[test]
    fn deploy_mode_must_match_compiled_target() {
        assert!(validate_deploy_mode(DeployMode::Host, DeployMode::Host).is_ok());
        assert!(matches!(
            validate_deploy_mode(DeployMode::SoC, DeployMode::Host),
            Err(Error::UnsupportedDeployment(DeployMode::SoC))
        ));
    }
}
