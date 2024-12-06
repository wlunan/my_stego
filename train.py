"""
@Time  : 2024/12/5 21:53       
@Author: xiaonan
@Email : itjunhao@qq.com
@File  : train.py        
"""
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import model
import os
from PIL import Image
class BatchSplitDataset(Dataset):
    def __init__(self, root_dir, batch_size, transform=None):
        self.root_dir = root_dir
        self.images = sorted(os.listdir(root_dir))
        self.batch_size = batch_size
        self.transform = transform

        if len(self.images) % batch_size != 0:
            raise ValueError("Total number of images must be divisible by the batch size.")

    def __len__(self):
        # 每个批次返回一个封面和秘密图像对
        return len(self.images) // self.batch_size

    def __getitem__(self, idx):
        # 定位到当前批次的起始索引
        start_idx = idx * self.batch_size
        batch_images = self.images[start_idx:start_idx + self.batch_size]

        # 将批次拆分为封面和秘密图像路径
        half_batch_size = self.batch_size // 2
        cover_paths = batch_images[:half_batch_size]
        secret_paths = batch_images[half_batch_size:]

        # 加载封面图像和秘密图像
        cover_imgs = [Image.open(os.path.join(self.root_dir, path)).convert("RGB") for path in cover_paths]
        secret_imgs = [Image.open(os.path.join(self.root_dir, path)).convert("RGB") for path in secret_paths]

        # 应用变换
        if self.transform:
            cover_imgs = [self.transform(img) for img in cover_imgs]
            secret_imgs = [self.transform(img) for img in secret_imgs]

        # 将封面图像和秘密图像拼接成 Tensor
        cover_tensor = torch.stack(cover_imgs)  # shape: [batch_size/2, 3, H, W]
        secret_tensor = torch.stack(secret_imgs)  # shape: [batch_size/2, 3, H, W]

        return cover_tensor, secret_tensor

def train_val_data_process():
    ROOT_TRAIN = 'E:\\01data\DIV2K_HR\\train'
    ROOT_VAL = 'E:/01data/DIV2K_HR/val'
    BATCH_SIZE = 2

    train_transform = transforms.Compose([
      transforms.Resize((256, 256)),transforms.ToTensor()
    ])
    # 加载数据集
    dataset = BatchSplitDataset(root_dir=ROOT_TRAIN, batch_size=BATCH_SIZE, transform=train_transform)
    # 划分训练集和验证集
    train_data, val_data = data.random_split(dataset,
                                             lengths=[round(0.8*len(dataset)), round(0.2*len(dataset))])
    train_loader = data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, val_loader

import torchvision.utils as vutils
import matplotlib.pyplot as plt

def train_model(hnet, rnet, train_loader, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义损失函数和优化器
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(hnet.parameters(), lr=0.001)

    hnet.to(device)
    rnet.to(device)

    # 训练参数
    epochs = 10
    tortal_secret_loss = 0.0
    secret_nums = 0
    start_time = time.time()
    # 开始训练
    hnet.train()
    rnet.train()

    for epoch in range(epochs):
        last_secret_img = None
        last_decoded_img = None

        for step, (cover_img, secret_img) in enumerate(train_loader):
            # 拼接图像
            combined = torch.cat((cover_img, secret_img), dim=2)
            combined = torch.squeeze(combined, dim=0)

            # 放入GPU
            combined = combined.to(device)
            secret_img = secret_img.to(device)

            # 前向传播
            encode = hnet(combined)
            decode = rnet(encode)

            # 计算原始秘密图像和解密图像之间的损失
            secret_loss = mse_loss(secret_img, decode)
            tortal_secret_loss += secret_loss.item()
            secret_nums += secret_img.size(0)
            # 反向传播和优化
            optimizer.zero_grad()
            secret_loss.backward()
            optimizer.step()

            # 保存最后一个批次的四个图像
            last_cover_img = cover_img.detach().cpu()
            last_secret_img = secret_img.detach().cpu()
            last_decoded_img = decode.detach().cpu()
            last_combined_img = combined.detach().cpu()

            end_time = time.time()

            # 输出当前 epoch 的训练损失
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {tortal_secret_loss / secret_nums :.4f}, Time: {end_time - start_time:.2f} s')

        # 可视化最后一个批次的四个图像
        visualize_images(last_cover_img, last_secret_img, last_combined_img, last_decoded_img, epoch)


def visualize_images(cover_imgs, secret_imgs, combined_imgs, decoded_imgs, epoch):
    """
    可视化四个图像：原始封面图像、原始秘密图像、结合图像和解密后的秘密图像。
    """
    # 确保每个输入图像是 4D 张量
    cover_imgs = torch.squeeze(cover_imgs, dim=0)
    secret_imgs = torch.squeeze(secret_imgs, dim=0)
    # combined_imgs = torch.squeeze(combined_imgs, dim=0)
    # decoded_imgs = torch.squeeze(decoded_imgs, dim=0)

    # 将 6 通道图像通过平均通道来压缩为 3 通道
    combined_imgs_rgb = combined_imgs.mean(dim=1, keepdim=False)

    # 创建图像网格
    cover_grid = vutils.make_grid(cover_imgs, nrow=2, normalize=True, scale_each=True)
    secret_grid = vutils.make_grid(secret_imgs, nrow=2, normalize=True, scale_each=True)
    combined_grid = vutils.make_grid(combined_imgs_rgb, nrow=2, normalize=True, scale_each=True)
    decoded_grid = vutils.make_grid(decoded_imgs, nrow=2, normalize=True, scale_each=True)

    # 打印图像维度
    print(f"Cover Image Shape: {cover_imgs.shape}, Secret Image Shape: {secret_imgs.shape}")
    print(f"Combined Image Shape: {combined_imgs.shape}, Decoded Image Shape: {decoded_imgs.shape}")

    # 使用 Matplotlib 显示
    plt.figure(figsize=(15, 10))

    # 原始封面图像
    plt.subplot(2, 2, 1)
    plt.title(f'Epoch {epoch + 1} - Cover Image')
    plt.imshow(cover_grid.permute(1, 2, 0))  # 转换为 HWC 格式
    plt.axis('off')

    # 原始秘密图像
    plt.subplot(2, 2, 2)
    plt.title(f'Epoch {epoch + 1} - Secret Image')
    plt.imshow(secret_grid.permute(1, 2, 0))  # 转换为 HWC 格式
    plt.axis('off')

    # 结合的封面图像
    plt.subplot(2, 2, 3)
    plt.title(f'Epoch {epoch + 1} - Combined Image')
    plt.imshow(combined_grid.permute(1, 2, 0))  # 转换为 HWC 格式
    plt.axis('off')

    # 解密后的秘密图像
    plt.subplot(2, 2, 4)
    plt.title(f'Epoch {epoch + 1} - Decoded Secret Image')
    plt.imshow(decoded_grid.permute(1, 2, 0))  # 转换为 HWC 格式
    plt.axis('off')

    # 保存或显示图像
    plt.savefig(f'epoch_{epoch + 1}_images.png')
    plt.show()


if __name__ == '__main__':
    hnet = model.HidingNet()
    rnet = model.RevealNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    hnet = hnet.to(device)
    rnet = rnet.to(device)
    # 处理数据
    train_loader, val_loader = train_val_data_process()
    # 开始训练
    train_model(hnet, rnet, train_loader, val_loader)