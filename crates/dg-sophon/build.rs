use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=LIBSOPHON_ROOT");
    if env::var_os("CARGO_FEATURE_BACKEND").is_none() {
        return;
    }

    let sdk_root = env::var_os("LIBSOPHON_ROOT")
        .map(PathBuf::from)
        .expect("LIBSOPHON_ROOT must point to a Sophon SDK when the backend feature is enabled");
    let include_dir = sdk_root.join("include");
    if !include_dir.exists() {
        panic!("LIBSOPHON_ROOT does not contain an include directory");
    }

    let lib_dir = if sdk_root.join("lib").exists() {
        sdk_root.join("lib")
    } else if sdk_root.join("lib64").exists() {
        sdk_root.join("lib64")
    } else {
        panic!("LIBSOPHON_ROOT does not contain a lib or lib64 directory");
    };

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=bmrt");
    println!("cargo:rustc-link-lib=dylib=bmlib");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let wrapper = out_dir.join("sophon_wrapper.h");
    fs::write(
        &wrapper,
        "#include \"bmrt.h\"\n#include \"bmlib_runtime.h\"\n",
    )
    .expect("write wrapper header");

    let bindings = bindgen::Builder::default()
        .header(wrapper.to_string_lossy().into_owned())
        .clang_arg(format!("-I{}", include_dir.display()))
        .derive_default(true)
        .derive_eq(true)
        .allowlist_function("bmrt_.*")
        .allowlist_function("bm_.*")
        .allowlist_type("bm_.*")
        .allowlist_var("BM_.*")
        .generate()
        .expect("generate Sophon bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("write Sophon bindings");
}
