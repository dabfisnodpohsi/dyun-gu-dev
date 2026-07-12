use dg_core::{DeployMode, DeviceKind, MemoryDomain};
use dg_runtime::{backend_capabilities, BackendKind};
use dg_scheduler::Request;
use tracing::trace;

use crate::Result;

/// Handle representations understood by a frame producer or consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HandleKind {
    HostBytes,
    External,
    Avcodec,
    CheetahBytes,
}

/// Layout information required before two frame buffers can be shared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameLayout {
    pub dims: Vec<usize>,
    pub format: MemoryFormat,
    pub dtype: MemoryDtype,
    pub plane_count: usize,
    pub strides: Vec<usize>,
    pub subsampling: Option<Subsampling>,
    pub packed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryFormat {
    Auto,
    Gray8,
    Rgb24,
    Rgba,
    Yuv420p,
    Nv12,
    Packet,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryDtype {
    U8,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Subsampling {
    Yuv420,
    Yuv422,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransferMode {
    Shared,
    Staged,
}

/// Actual transfer decision and diagnostics for one bridge operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransferReport {
    pub source_domain: MemoryDomain,
    pub target_domain: MemoryDomain,
    pub path: CopyPath,
    pub copy_count: usize,
    pub mode: TransferMode,
}

/// Inputs needed to decide whether an external frame can be shared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameTransferRequest {
    pub source_domain: MemoryDomain,
    pub target_domain: MemoryDomain,
    pub source_handle: HandleKind,
    pub target_handle: HandleKind,
    pub source_layout: FrameLayout,
    pub target_layout: FrameLayout,
    pub has_lifetime_guard: bool,
    pub staging_supported: bool,
    pub operation: String,
}

/// Preferred target memory domain for a backend/device/deployment combination.
pub fn preferred_memory_domain(device: DeviceKind, deploy_mode: DeployMode) -> MemoryDomain {
    match (device, deploy_mode) {
        (DeviceKind::Cpu, _) | (DeviceKind::IntelGpu, _) | (DeviceKind::IntelNpu, _) => {
            MemoryDomain::Host
        }
        (DeviceKind::CudaGpu, _) => MemoryDomain::CudaDevice,
        (DeviceKind::RknnNpu, DeployMode::SoC) => MemoryDomain::MppBuffer,
        (DeviceKind::RknnNpu, DeployMode::Host) => MemoryDomain::DmaBuf,
        (DeviceKind::SophonTpu, DeployMode::SoC) => MemoryDomain::SophonDevice,
        (DeviceKind::SophonTpu, DeployMode::Host) => MemoryDomain::SophonDevice,
    }
}

/// Copy path selected by the planner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyPath {
    pub domains: Vec<MemoryDomain>,
    pub copy_count: usize,
}

/// Planning request for zero-copy evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZeroCopyRequest {
    pub backend: BackendKind,
    pub device: DeviceKind,
    pub deploy_mode: DeployMode,
    pub source_domain: MemoryDomain,
    pub affinity_key: Option<String>,
}

/// Result of zero-copy planning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZeroCopyPlan {
    pub backend: BackendKind,
    pub device: DeviceKind,
    pub deploy_mode: DeployMode,
    pub target_domain: MemoryDomain,
    pub path: CopyPath,
    pub zero_copy: bool,
    pub supported: bool,
}

/// Zero-copy planner that uses backend capability data and a scheduler request.
#[derive(Clone, Debug, Default)]
pub struct ZeroCopyPlanner;

impl ZeroCopyPlanner {
    pub fn new() -> Self {
        Self
    }

    pub fn plan(&self, request: &ZeroCopyRequest) -> Result<ZeroCopyPlan> {
        let Some(caps) = backend_capabilities(request.backend) else {
            return Err(dg_core::Error::Unsupported(format!(
                "unsupported backend: {:?}",
                request.backend
            )));
        };
        if !caps.supports_device(request.device) {
            return Err(dg_core::Error::Unsupported(format!(
                "unsupported device: {:?}",
                request.device
            )));
        }
        if !caps.supports_deployment(request.deploy_mode) {
            return Err(dg_core::Error::Unsupported(format!(
                "unsupported deployment: {:?}",
                request.deploy_mode
            )));
        }

        let target_domain = preferred_memory_domain(request.device, request.deploy_mode);
        let staged =
            if request.source_domain == MemoryDomain::Host || target_domain == MemoryDomain::Host {
                CopyPath {
                    domains: vec![request.source_domain, target_domain],
                    copy_count: 1,
                }
            } else {
                CopyPath {
                    domains: vec![request.source_domain, MemoryDomain::Host, target_domain],
                    copy_count: 2,
                }
            };
        // The legacy request does not carry handle or layout compatibility
        // evidence, so it must never claim zero-copy.
        let zero_copy = false;
        let path = staged;

        trace!(
            backend = ?request.backend,
            device = ?request.device,
            deploy_mode = ?request.deploy_mode,
            source_domain = ?request.source_domain,
            target_domain = ?target_domain,
            copy_count = path.copy_count,
            zero_copy,
            "planned media transfer path"
        );

        Ok(ZeroCopyPlan {
            backend: request.backend,
            device: request.device,
            deploy_mode: request.deploy_mode,
            target_domain,
            path,
            zero_copy,
            supported: true,
        })
    }

    /// Plans a frame transfer without treating domain equality as sufficient
    /// evidence for zero-copy compatibility.
    pub fn plan_frame(&self, request: &FrameTransferRequest) -> Result<TransferReport> {
        let compatible = request.source_domain == request.target_domain
            && compatible_handle_kinds(request.source_handle, request.target_handle)
            && request.source_layout == request.target_layout
            && request.has_lifetime_guard;
        if compatible {
            let path = CopyPath {
                domains: vec![request.source_domain],
                copy_count: 0,
            };
            return Ok(TransferReport {
                source_domain: request.source_domain,
                target_domain: request.target_domain,
                path,
                copy_count: 0,
                mode: TransferMode::Shared,
            });
        }
        if !request.staging_supported {
            return Err(dg_core::Error::Unsupported(format!(
                "{} cannot stage from {:?} to {:?}",
                request.operation, request.source_domain, request.target_domain
            )));
        }
        let path = if request.source_domain == MemoryDomain::Host
            || request.target_domain == MemoryDomain::Host
        {
            CopyPath {
                domains: vec![request.source_domain, request.target_domain],
                copy_count: 1,
            }
        } else {
            CopyPath {
                domains: vec![
                    request.source_domain,
                    MemoryDomain::Host,
                    request.target_domain,
                ],
                copy_count: 2,
            }
        };
        let report = TransferReport {
            source_domain: request.source_domain,
            target_domain: request.target_domain,
            copy_count: path.copy_count,
            path,
            mode: TransferMode::Staged,
        };
        trace!(
            operation = %request.operation,
            source_domain = ?report.source_domain,
            target_domain = ?report.target_domain,
            copy_count = report.copy_count,
            mode = ?report.mode,
            "planned frame transfer"
        );
        Ok(report)
    }

    pub fn plan_with_request(
        &self,
        request: &ZeroCopyRequest,
        scheduler_request: Request,
    ) -> Result<(ZeroCopyPlan, Request)> {
        let plan = self.plan(request)?;
        Ok((plan, scheduler_request))
    }
}

fn compatible_handle_kinds(source: HandleKind, target: HandleKind) -> bool {
    source == target
        || matches!(
            (source, target),
            (HandleKind::Avcodec, HandleKind::External)
                | (HandleKind::External, HandleKind::Avcodec)
                | (HandleKind::HostBytes, HandleKind::External)
                | (HandleKind::External, HandleKind::HostBytes)
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> FrameLayout {
        FrameLayout {
            dims: vec![2, 2, 3],
            format: MemoryFormat::Rgb24,
            dtype: MemoryDtype::U8,
            plane_count: 1,
            strides: vec![6],
            subsampling: None,
            packed: true,
        }
    }

    fn request(source: MemoryDomain, target: MemoryDomain) -> FrameTransferRequest {
        FrameTransferRequest {
            source_domain: source,
            target_domain: target,
            source_handle: HandleKind::External,
            target_handle: HandleKind::External,
            source_layout: layout(),
            target_layout: layout(),
            has_lifetime_guard: true,
            staging_supported: true,
            operation: "test transfer".to_string(),
        }
    }

    #[test]
    fn compatible_host_handles_are_shared() {
        let report = ZeroCopyPlanner::new()
            .plan_frame(&request(MemoryDomain::Host, MemoryDomain::Host))
            .expect("compatible host transfer");
        assert_eq!(report.mode, TransferMode::Shared);
        assert_eq!(report.copy_count, 0);
    }

    #[test]
    fn domain_crossings_are_staged_with_expected_counts() {
        let host_to_device = ZeroCopyPlanner::new()
            .plan_frame(&request(MemoryDomain::Host, MemoryDomain::CudaDevice))
            .expect("host to device transfer");
        assert_eq!(host_to_device.mode, TransferMode::Staged);
        assert_eq!(host_to_device.copy_count, 1);

        let device_to_host = ZeroCopyPlanner::new()
            .plan_frame(&request(MemoryDomain::CudaDevice, MemoryDomain::Host))
            .expect("device to host transfer");
        assert_eq!(device_to_host.copy_count, 1);

        let device_to_device = ZeroCopyPlanner::new()
            .plan_frame(&request(MemoryDomain::CudaDevice, MemoryDomain::MppBuffer))
            .expect("device to device transfer");
        assert_eq!(device_to_device.copy_count, 2);
        assert_eq!(
            device_to_device.path.domains,
            vec![
                MemoryDomain::CudaDevice,
                MemoryDomain::Host,
                MemoryDomain::MppBuffer
            ]
        );
    }

    #[test]
    fn mismatched_layout_or_missing_staging_is_not_shared() {
        let mut mismatched = request(MemoryDomain::Host, MemoryDomain::Host);
        mismatched.target_layout.packed = false;
        let report = ZeroCopyPlanner::new()
            .plan_frame(&mismatched)
            .expect("mismatched layout stages");
        assert_eq!(report.mode, TransferMode::Staged);
        assert_eq!(report.copy_count, 1);

        mismatched.staging_supported = false;
        let error = ZeroCopyPlanner::new()
            .plan_frame(&mismatched)
            .expect_err("missing staging must be explicit");
        assert!(error.to_string().contains("test transfer"));
        assert!(error.to_string().contains("Host"));
    }
}
