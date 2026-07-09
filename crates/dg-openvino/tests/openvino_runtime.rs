#![cfg(feature = "backend")]

use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use dg_core::{DataFormat, DataType, DeviceKind, Shape, Tensor, TensorDesc};
use dg_runtime::{
    BackendKind, BackendOptions, ModelSource, OpenVINOOptions, Runtime, RuntimeOption,
};

fn python_command() -> Command {
    for candidate in ["python", "python3"] {
        if Command::new(candidate).arg("--version").output().is_ok() {
            return Command::new(candidate);
        }
    }
    Command::new("python")
}

fn unique_temp_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be monotonic")
        .as_nanos();
    std::env::temp_dir().join(format!("dg-openvino-{nanos}-{}", std::process::id()))
}

fn create_identity_model(output_dir: &Path) -> PathBuf {
    let model_path = output_dir.join("identity.xml");
    let script = r#"
import numpy as np
import openvino as ov
from openvino import opset8 as ops
import sys

target = sys.argv[1]
param = ops.parameter([1, 4], dtype=np.float32, name='input')
result = ops.result(param, name='output')
model = ov.Model([result], [param], 'identity')
ov.save_model(model, target)
"#;

    let status = python_command()
        .arg("-c")
        .arg(script)
        .arg(&model_path)
        .status()
        .expect("python should be available to build the model");
    assert!(status.success(), "python OpenVINO model generation failed");
    assert!(model_path.exists(), "XML model should exist");
    assert!(
        model_path.with_extension("bin").exists(),
        "BIN weights should exist"
    );
    model_path
}

fn openvino_lib_dir() -> PathBuf {
    let script = r#"
import pathlib
import openvino
print(pathlib.Path(openvino.__file__).resolve().parent / 'libs')
"#;

    let output = python_command()
        .arg("-c")
        .arg(script)
        .output()
        .expect("python should be available to locate OpenVINO libs");
    assert!(
        output.status.success(),
        "failed to discover OpenVINO library directory"
    );
    let path = String::from_utf8(output.stdout).expect("OpenVINO lib path should be UTF-8");
    PathBuf::from(path.trim())
}

fn prepare_loader_dir(root: &Path) -> PathBuf {
    let loader_dir = root.join("loader");
    std::fs::create_dir_all(&loader_dir).expect("create loader dir");
    let lib_dir = openvino_lib_dir();
    for (link_name, target_name) in [
        ("libopenvino.so", "libopenvino.so.2621"),
        ("libopenvino_c.so", "libopenvino_c.so.2621"),
    ] {
        let link = loader_dir.join(link_name);
        if link.exists() {
            std::fs::remove_file(&link).expect("remove stale loader symlink");
        }
        symlink(lib_dir.join(target_name), &link).expect("create OpenVINO loader symlink");
    }
    loader_dir
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn run_openvino_identity_model(model_path: PathBuf) {
    let option = RuntimeOption::new(
        BackendKind::OpenVINO,
        ModelSource::File(model_path),
        BackendOptions::OpenVINO(OpenVINOOptions::default()),
    )
    .with_precision(DataType::F32)
    .with_device(DeviceKind::Cpu);

    let mut runtime = Runtime::new(option).expect("construct OpenVINO runtime");
    assert_eq!(runtime.input_count(), 1);
    assert_eq!(runtime.output_count(), 1);

    let device = dg_core::CpuDevice::new();
    let input_desc = TensorDesc::new(
        Shape::new([1, 4]),
        DataType::F32,
        DataFormat::NC,
        DeviceKind::Cpu,
    )
    .with_name("input");
    let input = Tensor::allocate(&device, input_desc).expect("allocate input");
    let input_values = [1.0f32, -2.0, 3.5, 7.25];
    input
        .buffer()
        .write_from_slice(&f32_bytes(&input_values))
        .expect("seed input tensor");

    let outputs = runtime.run(&[input]).expect("run OpenVINO backend");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].buffer().read_bytes(), f32_bytes(&input_values));
}

#[test]
#[ignore]
fn openvino_identity_model_runs_end_to_end() {
    if std::env::var_os("DG_OPENVINO_E2E_CHILD").is_some() {
        let model_path = std::env::var_os("DG_OPENVINO_E2E_MODEL_PATH")
            .map(PathBuf::from)
            .expect("model path should be provided");
        run_openvino_identity_model(model_path);
        return;
    }

    assert!(dg_openvino::backend_enabled());

    let temp_dir = unique_temp_dir();
    std::fs::create_dir_all(&temp_dir).expect("create temp dir");
    let loader_dir = prepare_loader_dir(&temp_dir);
    let lib_dir = openvino_lib_dir();
    let model_path = create_identity_model(&temp_dir);

    let current_exe = std::env::current_exe().expect("locate current test binary");
    let current_ld_library_path = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let ld_library_path = format!(
        "{}:{}:{}",
        loader_dir.display(),
        lib_dir.display(),
        current_ld_library_path
    );

    let status = Command::new(current_exe)
        .arg("--exact")
        .arg("openvino_identity_model_runs_end_to_end")
        .arg("--ignored")
        .arg("--nocapture")
        .env("DG_OPENVINO_E2E_CHILD", "1")
        .env("DG_OPENVINO_E2E_MODEL_PATH", &model_path)
        .env("LD_LIBRARY_PATH", ld_library_path)
        .status()
        .expect("spawn OpenVINO child test process");
    assert!(status.success(), "child OpenVINO process should succeed");
}
