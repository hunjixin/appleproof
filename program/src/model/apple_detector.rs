// Generated from ONNX "src/model/apple_detector.onnx" by burn-import
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig2d;
use burn::record::BinBytesRecorder;
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;
use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv2d1: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    conv2d2: Conv2d<B>,
    maxpool2d2: MaxPool2d,
    gemm1: Linear<B>,
    gemm2: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}


static EMBEDDED_STATES: &[u8] =
    include_bytes!("/Users/waylon/code/appleproof/program/src/model/apple_detector.bin");

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_embedded(&Default::default())
    }
}

impl<B: Backend> Model<B> {
    pub fn from_embedded(device: &B::Device) -> Self {
        let record = BinBytesRecorder::<FullPrecisionSettings, &'static [u8]>::default()
            .load(EMBEDDED_STATES, device)
            .expect("Should decode state successfully");
        Self::new(device).load_record(record)
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 16], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d1 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d2 = Conv2dConfig::new([16, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d2 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let gemm1 = LinearConfig::new(32768, 64).with_bias(true).init(device);
        let gemm2 = LinearConfig::new(64, 1).with_bias(true).init(device);
        Self {
            conv2d1,
            maxpool2d1,
            conv2d2,
            maxpool2d2,
            gemm1,
            gemm2,
            phantom: core::marker::PhantomData,
            device: burn::module::Ignored(device.clone()),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input1: Tensor<B, 4>) -> Tensor<B, 2> {
        let conv2d1_out1 = self.conv2d1.forward(input1);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(relu1_out1);
        let conv2d2_out1 = self.conv2d2.forward(maxpool2d1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let maxpool2d2_out1 = self.maxpool2d2.forward(relu2_out1);
        let flatten1_out1 = {
            let leading_dim = maxpool2d2_out1.shape().dims[..1].iter().product::<usize>() as i32;
            maxpool2d2_out1.reshape::<2, _>([leading_dim, -1])
        };
        let gemm1_out1 = self.gemm1.forward(flatten1_out1);
        let relu3_out1 = burn::tensor::activation::relu(gemm1_out1);
        let gemm2_out1 = self.gemm2.forward(relu3_out1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(gemm2_out1);
        sigmoid1_out1
    }
}
