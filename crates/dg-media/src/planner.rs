use dg_core::{DeployMode, DeviceKind, MemoryDomain};
use dg_runtime::{backend_capabilities, BackendKind};
use dg_scheduler::Request;
use tracing::trace;

use crate::Result;

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
        let zero_copy = request.source_domain == target_domain;
        let path = if zero_copy {
            CopyPath {
                domains: vec![request.source_domain],
                copy_count: 0,
            }
        } else if request.source_domain == MemoryDomain::Host || target_domain == MemoryDomain::Host
        {
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

    pub fn plan_with_request(
        &self,
        request: &ZeroCopyRequest,
        scheduler_request: Request,
    ) -> Result<(ZeroCopyPlan, Request)> {
        let plan = self.plan(request)?;
        Ok((plan, scheduler_request))
    }
}
