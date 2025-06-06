//! An end-to-end example of using the SP1 SDK to generate a proof of a program that can be executed
//! or have a core proof generated.
//!
//! You can run this script using the following command:
//! ```shell
//! RUST_LOG=info cargo run --release -- --execute
//! ```
//! or
//! ```shell
//! RUST_LOG=info cargo run --release -- --prove
//! ```
use burn::tensor::TensorData;
use clap::Parser;
use image::imageops::FilterType;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::path::Path;

/// The ELF (executable and linkable format) file for the Succinct RISC-V zkVM.
pub const APPLEPROOF_ELF: &[u8] = include_elf!("appleproof-program");

/// The arguments for the command.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    execute: bool,

    #[arg(long)]
    prove: bool,

    #[arg(long, default_value = "", help = "Path to image")]
    path: String,
}

fn main() {
    // Setup the logger.
    sp1_sdk::utils::setup_logger();
    dotenv::dotenv().ok();

    // Parse the command line arguments.
    let args = Args::parse();

    if args.execute == args.prove {
        eprintln!("Error: You must specify either --execute or --prove");
        std::process::exit(1);
    }

    // Setup the prover client.
    let client = ProverClient::from_env();

    // Setup the inputs.
    let image_slice = preprocess_image(&args.path);
    let mut stdin = SP1Stdin::new();
    stdin.write_slice(&image_slice);

    if args.execute {
        // Execute the program
        let (output, report) = client.execute(APPLEPROOF_ELF, &stdin).run().unwrap();
        println!("Program executed successfully.");

        // Read the output.
        let is_apple = output.as_slice()[0] == 1;
        println!("is_apple: {}", is_apple);

        let expected_is_apple = appleproof_lib::has_apple(image_slice);
        assert_eq!(is_apple, expected_is_apple);
        if is_apple {
            println!("Is apple !");
        } else {
            println!("Not apple !");
        }

        // Record the number of cycles executed.
        println!("Number of cycles: {}", report.total_instruction_count());
    } else {
        // Setup the program for proving.
        let (pk, vk) = client.setup(APPLEPROOF_ELF);

        // Generate the proof
        let proof = client
            .prove(&pk, &stdin)
            .run()
            .expect("failed to generate proof");

        println!("Successfully generated proof!");

        // Verify the proof.
        client.verify(&proof, &vk).expect("failed to verify proof");
        println!("Successfully verified proof!");
    }
}

// 归一化参数
const MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const STD: [f32; 3] = [0.5, 0.5, 0.5];

// 读取图片，调整大小，归一化，转换成 burn tensor
pub fn preprocess_image(image_path: &str) -> Vec<u8> {
    // 1. 读取图片并转为 RGB
    let img = image::open(Path::new(image_path))
        .expect("Failed to open image")
        .to_rgb8();

    // 2. 调整大小到 128x128（双线性插值）
    let resized = image::imageops::resize(&img, 128, 128, FilterType::Triangle);

    // 3. 准备一个形状为 (3, 128, 128) 的 f32 数组（Burn 期望 channel-first）
    let mut data = vec![0f32; 3 * 128 * 128];

    for (x, y, pixel) in resized.enumerate_pixels() {
        // 归一化到 [0,1]
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        // 标准化 (channel-first)
        data[(y as usize) * 128 + (x as usize)] = (r - MEAN[0]) / STD[0];
        data[128 * 128 + (y as usize) * 128 + (x as usize)] = (g - MEAN[1]) / STD[1];
        data[2 * 128 * 128 + (y as usize) * 128 + (x as usize)] = (b - MEAN[2]) / STD[2];
    }

    // 4. 从数据创建 Tensor，维度 (1, 3, 128, 128)
    let tensor_data = TensorData::new(data, [1, 3, 128, 128]);
    tensor_data.as_bytes().to_vec()
}
