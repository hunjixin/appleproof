import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import sys

# -------------------------------
# 图像预处理（与训练时保持一致）
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# -------------------------------
# 加载图像并预处理
# -------------------------------
def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 128, 128]
    return tensor.numpy()

# -------------------------------
# 使用 ONNX Runtime 推理
# -------------------------------
def predict_onnx(image_path, model_path="apple_detector.onnx"):
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_tensor = preprocess(image_path)
    outputs = session.run([output_name], {input_name: input_tensor})
    prob = 1 / (1 + np.exp(-outputs[0]))  # Sigmoid 后处理
    return float(prob[0][0])

# -------------------------------
# 脚本入口
# -------------------------------
if __name__ == "__main__":
    img_path = sys.argv[1]
    print("load image ", img_path)
    probability = predict_onnx(img_path)
    print(f"🍎 Apple Probability (ONNX): {probability:.2%}")
