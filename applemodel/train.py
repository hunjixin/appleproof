import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# 设备选择
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU (MPS)")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")

print(f"Using device: {device}")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载数据集
full_dataset = datasets.ImageFolder("dataset", transform=transform)
class_names = full_dataset.classes
print("Classes:", class_names)

# ✅ 获取 'apple' 的标签索引
apple_index = class_names.index('apple')
print(f"'apple' class index: {apple_index}")

# 数据划分
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 模型定义
class ApplePresenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# 初始化
model = ApplePresenceClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# 训练循环
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        # ✅ 将标签重映射为是否为 'apple'（1）或不是（0）
        y = (y == apple_index).float().unsqueeze(1)
        x, y = x.to(device), y.to(device)
        
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            y = (y == apple_index).float().unsqueeze(1)
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Acc = {acc:.2%}")

# 导出 ONNX 模型
dummy_input = torch.randn(1, 3, 128, 128).to(device)
torch.onnx.export(model, dummy_input, "apple_detector.onnx",
                  input_names=["input"],
                  output_names=["output"],
                  opset_version=11)
print("Exported model to apple_detector.onnx")
