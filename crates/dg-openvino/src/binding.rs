use dg_core::MemoryDomain;

/// Binding route selected from the source memory domain and OpenVINO support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalBindingPath {
    HostTensor,
    RemoteTensor,
    Staging,
}

/// Selects the OpenVINO tensor route without inspecting vendor SDK state.
pub const fn select_external_binding_path(domain: MemoryDomain) -> ExternalBindingPath {
    match domain {
        MemoryDomain::Host => ExternalBindingPath::HostTensor,
        MemoryDomain::CudaDevice
        | MemoryDomain::DmaBuf
        | MemoryDomain::DrmPrime
        | MemoryDomain::VaapiSurface
        | MemoryDomain::MppBuffer
        | MemoryDomain::SophonDevice
        | MemoryDomain::Opaque => ExternalBindingPath::RemoteTensor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_selection_prefers_host_and_remote_tensor_routes() {
        assert_eq!(
            select_external_binding_path(MemoryDomain::Host),
            ExternalBindingPath::HostTensor
        );
        assert_eq!(
            select_external_binding_path(MemoryDomain::CudaDevice),
            ExternalBindingPath::RemoteTensor
        );
    }
}
