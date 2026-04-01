import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# 设置随机种子
torch.manual_seed(42)

# 数据预处理
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # Fashion-MNIST的均值和标准差
    ]
)

# 加载完整数据集（全量）
full_train = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
full_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

# 将训练集划分为训练集和验证集 TO DO


# 创建DataLoader
BATCH_SIZE = 64
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(full_test, batch_size=BATCH_SIZE)

print(f"训练样本数: {len(train_set)}, 验证样本数: {len(val_set)}, 测试样本数: {len(full_test)}")
print(f"类别标签: {full_train.classes}")
print(f"类别数量: {len(full_train.classes)}")

# 显示样本信息
sample_image, sample_label = full_train[0]
print(f"\n图像形状: {sample_image.shape}")
print(f"像素值范围: [{sample_image.min():.3f}, {sample_image.max():.3f}]")
print(f"示例标签: {sample_label} -> {full_train.classes[sample_label]}")

