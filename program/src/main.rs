//! A simple program that takes a number `n` as input, and writes the `n-1`th and `n`th fibonacci
//! number as an output.

// These two lines are necessary for the program to properly compile.
//
// Under the hood, we wrap your main function with some extra code so that it behaves properly
// inside the zkVM.
#![no_std]
#![no_main]

sp1_zkvm::entrypoint!(main);

pub mod model;
use burn::{backend::NdArray, tensor::Tensor};
use embedded_alloc::LlffHeap as Heap;
use model::Model;

type Backend = NdArray<f32>;
type BackendDevice = <Backend as burn::tensor::backend::Backend>::Device;

#[global_allocator]
static HEAP: Heap = Heap::empty();

pub fn main() {
    {
        use core::mem::MaybeUninit;
        const HEAP_SIZE: usize = 100 * 1024;
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        unsafe { HEAP.init(&raw mut HEAP_MEM as usize, HEAP_SIZE) }
    }
    // Get a default device for the backend
    let device = BackendDevice::default();

    // Create a new model and load the state
    let model: Model<Backend> = Model::default();

    // Define input
    let input = 0.0;

    // Run the model
    let output = run_model(&model, &device, input);

    // Output the values
    let output_data = output.into_data();
    let slice = output_data.as_slice::<f32>().unwrap();
    let data = slice[0].to_le_bytes();
    sp1_zkvm::io::commit_slice(&data);
}

fn run_model<'a>(model: &Model<NdArray>, device: &BackendDevice, input: f32) -> Tensor<Backend, 2> {
    // Define the tensor
    let input = Tensor::<Backend, 4>::from_floats([[input]], &device);

    // Run the model on the input
    let output = model.forward(input);

    output
}
