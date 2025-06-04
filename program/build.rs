//! This build script copies the `memory.x` file from the crate root into
//! a directory where the linker can always find it at build time.
//! For many projects this is optional, as the linker always searches the
//! project root directory -- wherever `Cargo.toml` is. However, if you
//! are using a workspace or have a more complicated build setup, this
//! build script becomes required. Additionally, by requesting that
//! Cargo re-run the build script whenever `memory.x` is changed,
//! updating `memory.x` ensures a rebuild of the application with the
//! new memory settings.

use burn_import::burn::graph::RecordType;
use burn_import::onnx::ModelGen;
use std::env;
use std::path::PathBuf;

fn main() {
    // Put `memory.x` in our output directory and ensure it's
    // on the linker search path.
    let out = &PathBuf::from(env::var_os("OUT_DIR").unwrap());
    println!("cargo:rustc-link-search={}", out.display());
    generate_model();
}

fn generate_model() {
    // Generate the model code from the ONNX file.
    ModelGen::new()
        .input("src/model/apple_detector.onnx")
        .out_dir("/Users/waylon/code/appleproof/program/src/model/")
        .record_type(RecordType::Bincode)
        .embed_states(true)
        .run_from_script();
}
