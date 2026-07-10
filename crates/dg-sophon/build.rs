use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=LIBSOPHON_ROOT");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_SOC");
    if env::var_os("CARGO_FEATURE_BACKEND").is_none() {
        return;
    }

    // SoC and Host (PCIe) targets ship the runtime under different sysroots but
    // expose the same `libbmrt`/`libbmlib` names; the deployment split is
    // enforced at runtime via the `soc` feature. Point `LIBSOPHON_ROOT` at the
    // matching SDK for the target being built.
    let soc = env::var_os("CARGO_FEATURE_SOC").is_some();

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
    // Surface the resolved deployment target so mismatched SDK roots are easy to
    // spot in build logs; the runtime guard rejects mode mismatches.
    println!(
        "cargo:warning=building dg-sophon for {} deployment",
        if soc { "SoC" } else { "Host (PCIe)" }
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let wrapper = out_dir.join("sophon_wrapper.h");
    fs::write(
        &wrapper,
        "#include \"bmruntime_interface.h\"\n#include \"bmlib_runtime.h\"\n",
    )
    .expect("write wrapper header");

    let bindings = bindgen::Builder::default()
        .header(wrapper.to_string_lossy().into_owned())
        .clang_arg(format!("-I{}", include_dir.display()))
        .derive_default(true)
        .derive_eq(true)
        .rustified_enum("bm_status_t")
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
