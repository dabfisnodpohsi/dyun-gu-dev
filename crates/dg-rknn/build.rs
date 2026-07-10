use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=RKNN_SDK_ROOT");
    if env::var_os("CARGO_FEATURE_BACKEND").is_none() {
        return;
    }

    let sdk_root = env::var_os("RKNN_SDK_ROOT").map(PathBuf::from).expect(
        "RKNN_SDK_ROOT must point to a Rockchip RKNN SDK when the backend feature is enabled",
    );
    let include_dir = sdk_root.join("include");
    let header = include_dir.join("rknn_api.h");
    if !header.exists() {
        panic!("RKNN_SDK_ROOT does not contain include/rknn_api.h");
    }

    let lib_dir = if sdk_root.join("lib").exists() {
        sdk_root.join("lib")
    } else if sdk_root.join("lib64").exists() {
        sdk_root.join("lib64")
    } else {
        panic!("RKNN_SDK_ROOT does not contain a lib or lib64 directory");
    };

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=rknnrt");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let wrapper = out_dir.join("rknn_wrapper.h");
    fs::write(&wrapper, "#include \"rknn_api.h\"\n").expect("write wrapper header");

    let bindings = bindgen::Builder::default()
        .header(wrapper.to_string_lossy().into_owned())
        .clang_arg(format!("-I{}", include_dir.display()))
        .derive_default(true)
        .derive_eq(true)
        .rustified_enum("_?rknn_query_cmd")
        .rustified_enum("_?rknn_tensor_type")
        .rustified_enum("_?rknn_tensor_format")
        .rustified_enum("_?rknn_tensor_qnt_type")
        .rustified_enum("_?rknn_core_mask")
        .allowlist_function("rknn_.*")
        .allowlist_type("rknn_.*")
        .allowlist_var("RKNN_.*")
        .generate()
        .expect("generate RKNN bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("write RKNN bindings");
}
