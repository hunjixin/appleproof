mod model;

use burn::{
    backend::NdArray,
    tensor::{activation::sigmoid, Tensor, TensorData},
};
use model::Model;

type Backend = NdArray<f32>;
type BackendDevice = <Backend as burn::tensor::backend::Backend>::Device;

pub fn has_apple(image_slice: Vec<u8>) -> bool {
    let device = BackendDevice::default();
    let model: Model<Backend> = Model::default();
    let output = run_model(&model, &device, image_slice);
    let prob_vec: Vec<f32> = sigmoid(output)
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap(); // apply sigmoid to logit
    prob_vec[0] > 0.8
}

fn run_model(model: &Model<NdArray>, device: &BackendDevice, input: Vec<u8>) -> Tensor<Backend, 2> {
    // Define the tensor
    let tensor_data = TensorData::from_bytes(input, [1, 3, 128, 128], burn::tensor::DType::F32);
    let input = Tensor::<Backend, 4>::from_data(tensor_data, device);
    model.forward(input)
}
