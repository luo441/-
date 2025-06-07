import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import os

# -------------------- CUDA加速配置 --------------------
torch.backends.cudnn.benchmark = True  # 启用cuDNN自动优化器
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -------------------- 配置文件路径 --------------------
DATASET_DIR = r"E:\Python\Project\ResNet\cifar-10-python\cifar-10-batches-py\cifar-10-batches"  
MODEL_SAVE_PATH = os.path.join(DATASET_DIR, "cifar10_resnet18.pth")

# -------------------- 自定义数据集加载器 --------------------
class CIFAR10Local(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []

        files = [f"data_batch_{i}" for i in range(1,6)] if train else ["test_batch"]
        
        for filename in files:
            filepath = os.path.join(root_dir, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"数据文件缺失: {filename}")
            
            with open(filepath, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                self.data.append(batch[b'data'])
                self.labels.extend(batch[b'labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------- 数据预处理管道 --------------------
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                        std=[0.2023, 0.1994, 0.2010])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010])
])

# -------------------- 适配CIFAR的ResNet-18 --------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------------------- 训练流程 --------------------
def main():
    # 初始化数据集
    try:
        train_set = CIFAR10Local(DATASET_DIR, train=True, transform=train_transform)
        test_set = CIFAR10Local(DATASET_DIR, train=False, transform=test_transform)
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        print("请检查以下文件是否存在:")
        print(f"路径: {DATASET_DIR}")
        print("[data_batch_1, ..., data_batch_5, test_batch]")
        return

    # 划分训练集和验证集
    train_size = int(0.9 * len(train_set))
    train_subset, val_subset = random_split(train_set, [train_size, len(train_set)-train_size])

    # 创建数据加载器
    BATCH_SIZE = 256
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    
    # 训练配置
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # 初始化记录器
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # 训练循环
    best_acc = 0.0
    for epoch in range(200):
        model.train()
        total_loss, correct = 0.0, 0
        
        # 训练阶段
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / len(train_subset)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                val_correct += outputs.argmax(1).eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / len(val_subset)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        # 更新学习率
        scheduler.step()
        
        print(f"Epoch {epoch+1}/200 | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    # 最终测试
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total += labels.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
    
    print(f"\n测试准确率: {100*correct/total:.2f}%")

    # 绘制性能曲线
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Windows多进程支持
    from multiprocessing import freeze_support
    freeze_support()
    
    # 启动训练
    main()