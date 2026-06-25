import torch
# x = torch.randn(3, 4)
# print(x)
# print(x.shape)
# print(x.device)

# # 1. 创建
# a = torch.zeros(2, 3)        # 全 0
# b = torch.ones(2, 3)         # 全 1
# c = torch.randn(2, 3) * 0.1  # 正态随机

# # 2. 索引和变形
# print(c[0, 1])               # 取第 0 行第 1 列
# print(c[:, 1])               # 取第 1 列全部
# print(c.view(6))             # 展平成一维
# print(c.view(3, 2))          # 改成 3×2

# # 3. 运算
# print(a + b)                 # 全是 1
# print(c @ c.T)               # 矩阵乘法 (2×3) × (3×2) = 2×2
# print(c.mean(), c.std())     # 均值、标准差

# 
# # 4. autograd：自动求导
# x = torch.tensor(2.0, requires_grad=True)
# y = x ** 2
# y.backward()
# print(x.grad)        # dy/dx = 2x = 4.0 ✅

# # 试一个复杂点的
# x = torch.tensor(3.0, requires_grad=True)
# y = x ** 3 + 2 * x + 1
# y.backward()
# print(x.grad)        # 手算：3x² + 2 = 29

# # 再来：多变量
# a = torch.tensor(2.0, requires_grad=True)
# b = torch.tensor(3.0, requires_grad=True)
# z = a**2 + 3 * a * b + b**2
# z.backward()
# print(a.grad)        # 手算：2a + 3b = 13
# print(b.grad)        # 手算：3a + 2b = 12

# 5.这一行导入了三个工具：
# datasets  → 里面自带 MNIST、CIFAR 等经典数据集，不用自己下载
# transforms → 图片预处理工具包
# DataLoader → 分批器
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── 第 1 步：定义「怎么处理每张图片」 ──
# transforms.Compose([...]) 的意思是「按顺序执行下面这些操作」
transform = transforms.Compose([
    # ToTensor()：把 PIL 图片（0~255 的整数）转成 tensor（0~1 的小数）
    transforms.ToTensor(),
    # Normalize(均值, 标准差)：让数据变成以 0 为中心、波动范围固定
    # 0.1307 和 0.3081 是 MNIST 统计出来的经验值，直接用就行
    transforms.Normalize((0.1307,), (0.3081,))
])

# ── 第 2 步：下载/加载 MNIST 数据集 ──
# MNIST 是手写数字 0~9，灰白图，28×28 像素，共 6 万张
# root='./data'：数据存到当前文件夹下的 data/ 目录
# train=True：要训练集（6 万张）
# download=True：本地没有就自动下载
# transform=transform：用上面定义的处理方式
train_data = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

# train=False：要测试集（1 万张），用来评估模型
test_data = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# ── 第 3 步：创建分批器 ──
# batch_size=64：每次拿出 64 张图一起处理（不是一张一张来）
# shuffle=True：每次训练前把牌洗乱，防止顺序影响学习
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# ── 第 4 步：取一批出来看看长什么样 ──
# iter(train_loader) 创建一个迭代器
# next(...) 取下一批
images, labels = next(iter(train_loader))

# images.shape → torch.Size([64, 1, 28, 28])
#   ↑ 含义：64张图，每张1个通道（灰度），高28，宽28
# labels.shape → torch.Size([64])
#   ↑ 含义：64个标签（0~9 的数字）
print(images.shape)    # torch.Size([64, 1, 28, 28]) — 64张图, 1通道, 28×28
print(labels.shape)    # torch.Size([64]) — 64个标签
print(labels[:10])     # 前10个标签