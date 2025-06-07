import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import ssl  # 解决SSL证书问题

# 1. 解决数据集下载时的SSL证书问题
ssl._create_default_https_context = ssl._create_unverified_context

# 2. 数据准备（增加错误重试机制）
def prepare_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 创建数据目录（避免权限问题）
    data_dir = './minist/data_mnist'  # 修改路径避免与系统冲突
    os.makedirs(data_dir, exist_ok=True)
    
    # 自动重试下载
    max_retries = 3
    for attempt in range(max_retries):
        try:
            train_val_data = datasets.MNIST(
                root=data_dir, 
                train=True, 
                download=True, 
                transform=transform
            )
            test_data = datasets.MNIST(
                root=data_dir, 
                train=False, 
                download=True, 
                transform=transform
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to download MNIST after {max_retries} attempts: {e}")
            print(f"Download failed (attempt {attempt + 1}), retrying...")
    
    # 划分数据集
    train_size = int(0.8 * len(train_val_data))
    val_size = len(train_val_data) - train_size
    train_data, val_data = random_split(train_val_data, [train_size, val_size])
    
    # 数据加载器
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)  # num_workers=0避免多进程问题
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

# 3. LeNet模型（兼容PyTorch 2.5+）
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 添加padding保持尺寸
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # 修正维度计算错误
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 4. 验证函数
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(dataloader), 100 * correct / total

# 5. 主流程（增加CUDA内存清理）
def main():
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 准备TensorBoard
    log_dir = './minist/runs/mnist_lenet_final'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 加载数据
    try:
        train_loader, val_loader, test_loader = prepare_data()
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    
    # 初始化模型
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    epochs = 15
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            
        
        # 计算指标
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 记录数据
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), './minist/best_lenet_mnist_final.pth')
        # 打印日志
        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # TensorBoard记录
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
    
    # 测试阶段
    try:
        model.load_state_dict(torch.load('./minist/best_lenet_mnist_final.pth', map_location=device))
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    except Exception as e:
        print(f"Test failed: {e}")
    # try:
    #     model.load_state_dict(torch.load('best_lenet_mnist_final.pth', map_location=device))
    #     test_loss, test_acc = validate(model, test_loader, criterion, device)
    #     print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
        
    #     # ===== 新增：打印最优模型参数内容 =====
    #     print("\n===== 最优模型参数详情 =====")
    #     state_dict = model.state_dict()
        
    #     # 打印各层参数形状和统计信息
    #     for name, param in state_dict.items():
    #         print(f"\n层名称: {name}")
    #         print(f"参数形状: {param.shape}")
    #         print(f"参数类型: {param.dtype}")
    #         print(f"最小值: {param.min().item():.6f}")
    #         print(f"最大值: {param.max().item():.6f}")
    #         print(f"平均值: {param.mean().item():.6f}")
    #         print(f"标准差: {param.std().item():.6f}")
            
    #         # 打印前5个参数值（避免输出过多）
    #         if param.numel() > 5:
    #             print("前5个参数值:", param.flatten()[:5].tolist())
    #         else:
    #             print("参数值:", param.tolist())
        
    #     # 打印总参数数量
    #     total_params = sum(p.numel() for p in model.parameters())
    #     print(f"\n模型总参数数量: {total_params:,}")
    #     # ===== 新增结束 =====
        


    # except Exception as e:
    #     print(f"Test failed: {e}")

    
    # 可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./minist/mnist_training_curves.png', dpi=300)
    plt.show()
    writer.close()
    
    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()