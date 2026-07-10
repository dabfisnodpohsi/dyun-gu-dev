//! Feature-gated integration scaffolding for the real Sophon backend.
//!
//! This test only compiles with `--features backend` (an SDK + `LIBSOPHON_ROOT`
//! are required to build). Even then it performs a real inference only when
//! `DG_SOPHON_TEST_BMODEL` points at a compiled bmodel and Sophon hardware is
//! present, so it is a no-op on machines without a device. It never fabricates
//! outputs.

#![cfg(feature = "backend")]

use std::path::PathBuf;

use dg_core::{DeployMode, DeviceKind};
use dg_runtime::{
    create_backend, BackendKind, BackendOptions, ModelSource, RuntimeOption, SophonOptions,
};
use dg_sophon::backend_enabled;

fn bmodel_path() -> Option<PathBuf> {
    std::env::var_os("DG_SOPHON_TEST_BMODEL").map(PathBuf::from)
}

#[test]
fn backend_is_enabled_when_compiled() {
    assert!(backend_enabled());
}

#[test]
fn discovers_metadata_from_real_bmodel() {
    let Some(path) = bmodel_path() else {
        eprintln!("skipping: set DG_SOPHON_TEST_BMODEL to a bmodel to run this test");
        return;
    };

    let option = RuntimeOption::new(
        BackendKind::Sophon,
        ModelSource::File(path),
        BackendOptions::Sophon(SophonOptions {
            deploy_mode: DeployMode::Host,
            device_id: Some(0),
            core_mask: None,
        }),
    )
    .with_device(DeviceKind::SophonTpu)
    .with_deploy_mode(DeployMode::Host);

    let mut backend = create_backend(BackendKind::Sophon).expect("Sophon backend is registered");
    backend
        .init(&option)
        .expect("initialize Sophon backend on hardware");

    assert!(backend.input_count() >= 1);
    assert!(backend.output_count() >= 1);
    for index in 0..backend.input_count() {
        let info = backend.input_info(index).expect("input info");
        assert!(info.shape.rank() >= 1);
    }
}
