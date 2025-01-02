"""
@Time  : 2024/12/5 21:53       
@Author: xiaonan
@Email : itjunhao@qq.com
@File  : train.py        
"""
import copy
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import models.model as model
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# 自己添加的模块
from datasets.data_loader import BatchSplitDataset
from configs.config import Config

def train_val_data_process():
    ROOT_TRAIN = Config.root_train
    ROOT_VAL = Config.root_val
    BATCH_SIZE = Config.batch_size

    train_transform = transforms.Compose([
      transforms.Resize((256, 256)),transforms.ToTensor()
    ])
    # 加载数据集
    dataset = BatchSplitDataset(root_dir=ROOT_TRAIN, transform=train_transform)
    # 划分训练集和验证集
    train_data, val_data = data.random_split(dataset,
                                             lengths=[round(0.8*len(dataset)), round(0.2*len(dataset))])

    train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader

# 计算 PSNR
def calculate_psnr(original, reconstructed, max_value=1.0):
    # 确保目标张量有批量维度
    if original.dim() == 3:  # 如果没有批量维度
        original = original.unsqueeze(0)

    # 计算均方误差 MSE
    mse = F.mse_loss(reconstructed, original)

    # 计算 PSNR
    psnr = 10 * torch.log10((max_value ** 2) / mse)

    return psnr.item()  # 返回 PSNR 值

def train_model(hnet, rnet, train_loader, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter("logs")
    # 最佳权重
    best_hnet_model = copy.deepcopy(hnet.state_dict)
    best_rnet_model = copy.deepcopy(rnet.state_dict)

    # 定义损失函数和优化器
    mse_fun = nn.MSELoss()

    r_optimizer = torch.optim.Adam(hnet.parameters(), lr=0.001)
    h_optimizer = torch.optim.Adam(rnet.parameters(), lr=0.001)

    hnet.to(device)
    rnet.to(device)

    # 训练参数
    epochs = Config.epochs

    total_hnet_mseloss = 0.0 # 隐藏网络的损失 封面图像和结合图像之间的损失
    cover_nums = 0
    total_rnet_mseloss = 0.0 # 秘密图像的损失 秘密图像和解密图像之间的损失
    secret_nums = 0
    sum_mseloss = 0.0 # 总的损失

    total_rnet_psnr = 0.0 # 一个epoch中的总PSNR
    secret_nums = 0

    start_time = time.time()
    min_sum_mseloss = 1

    # 开始训练
    hnet.train()
    rnet.train()

    for epoch in range(epochs):
        last_secret_img = None
        last_decoded_img = None
        total_psnr = 0.0
        count_psnr = 0

        for step, (cover_img, secret_img) in enumerate(train_loader):
            # print(cover_img.shape)
            # print(secret_img.shape)
            # 拼接图像
            combined = torch.cat((cover_img, secret_img), dim=0)
            # print(combined.shape)
            combined = torch.unsqueeze(combined, 0) # 增加一个维度

            # 放入GPU
            # cover_img = cover_img.to(device)
            # secret_img = secret_img.to(device)
            cover_img = cover_img.to(device).unsqueeze(0)  # 增加批量维度
            secret_img = secret_img.to(device).unsqueeze(0)  # 增加批量维度
            combined = combined.to(device)
            secret_img = secret_img.to(device)

            # 前向传播
            container_img = hnet(combined)
            rev_secret_img = rnet(container_img)

            # 计算损失
            hnet_loss = mse_fun(cover_img, container_img)
            rnet_loss = mse_fun(secret_img, rev_secret_img)
            # 添加总损失，平衡两个网络的损失，这样封面和揭示图像效果应该会更好，因为要是只用一个网络的话，另一个网络的损失会很大
            total_loss = hnet_loss + rnet_loss

            total_hnet_mseloss += hnet_loss.item()
            total_rnet_mseloss += rnet_loss.item()

            secret_nums += secret_img.size(0)
            cover_nums += cover_img.size(0)
            sum_mseloss += total_loss.item()

            # 计算当前批次的 PSNR，秘密图像和解密图像之间的 PSNR
            psnr_value = calculate_psnr(secret_img, rev_secret_img)
            total_psnr += psnr_value
            count_psnr += 1

            # 反向传播和优化
            h_optimizer.zero_grad()
            r_optimizer.zero_grad()

            # hnet_loss.backward()
            # rnet_loss.backward() # 因为是作为一个整体训练的，所以只需要反向传播一次
            total_loss.backward() # 使用总的损失反向传播

            h_optimizer.step()
            r_optimizer.step()

            # 保存最后一个批次的四个图像
            last_cover_img = cover_img.detach().cpu()
            last_secret_img = secret_img.detach().cpu()
            last_combined_img = container_img.detach().cpu()
            last_rev_secret_img = rev_secret_img.detach().cpu()

            # 保存模型 当前批次的损失小于最小损失
            if sum_mseloss < min_sum_mseloss:
                min_sum_mseloss = sum_mseloss
                best_hnet_model = copy.deepcopy(hnet.state_dict())
                best_rnet_model = copy.deepcopy(rnet.state_dict())

            # 记录训练损失
            writer.add_scalar("Hnet Loss", total_hnet_mseloss / cover_nums, epoch * len(train_loader) + step)
            writer.add_scalar("Rnet Loss", total_rnet_mseloss / secret_nums, epoch * len(train_loader) + step)
            writer.add_scalar("Train Loss", sum_mseloss, epoch * len(train_loader) + step)

            end_time = time.time()

        # 每十轮保存一次
        torch.save(best_hnet_model,
                   f'checkpoints/epoch_{epoch + 1}_sum_loss_{sum_mseloss:.5f}_hnet_loss_{hnet_loss:.5f}_rnet_loss_{rnet_loss:.5f}_best_hnet_model.pth')
        torch.save(best_rnet_model,
                   f'checkpoints/epoch_{epoch + 1}_sum_loss_{sum_mseloss:.5f}_hnet_loss_{hnet_loss:.5f}_rnet_loss_{rnet_loss:.5f}_best_rnet_model.pth')

        # 输出当前 epoch 的训练损失和 PSNR
        avg_psnr = total_psnr / count_psnr if count_psnr > 0 else 0
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_hnet_mseloss / secret_nums:.4f}, PSNR: {avg_psnr:.2f} dB, Time: {end_time - start_time:.2f} s')
        # print(f'Epoch [{epoch + 1}/{epochs}], Loss: {sum_mseloss / secret_nums:.4f} Time: {end_time - start_time:.2f} s')

        # 可视化最后一个批次的四个图像
        visualize_images(last_cover_img, last_secret_img, last_combined_img, last_rev_secret_img, epoch)


def visualize_images(cover_imgs, secret_imgs, combined_imgs, decoded_imgs, epoch):
    """
    可视化四个图像：原始封面图像、原始秘密图像、结合图像和解密后的秘密图像。
    """
    # # 确保每个输入图像是 4D 张量
    # cover_imgs = torch.squeeze(cover_imgs, dim=0)
    # secret_imgs = torch.squeeze(secret_imgs, dim=0)

    # 将 6 通道图像通过平均通道来压缩为 3 通道
    # combined_imgs_rgb = combined_imgs.mean(dim=1, keepdim=False)

    # 创建图像网格
    cover_grid = vutils.make_grid(cover_imgs, nrow=2, normalize=True, scale_each=True)
    secret_grid = vutils.make_grid(secret_imgs, nrow=2, normalize=True, scale_each=True)
    combined_grid = vutils.make_grid(combined_imgs, nrow=2, normalize=True, scale_each=True)
    decoded_grid = vutils.make_grid(decoded_imgs, nrow=2, normalize=True, scale_each=True)

    # 打印图像维度
    # print(f"Cover Image Shape: {cover_imgs.shape}, Secret Image Shape: {secret_imgs.shape}")
    # print(f"Combined Image Shape: {combined_imgs.shape}, Decoded Image Shape: {decoded_imgs.shape}")

    # 使用 Matplotlib 显示
    plt.figure(figsize=(20, 10))
    # 设置子图之间的间距
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
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
    plt.savefig(f'./results/epoch_{epoch + 1}_images.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    hnet = model.HidingNet()
    rnet = model.RevealNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(torch.cuda.is_available())
    hnet = hnet.to(device)
    rnet = rnet.to(device)
    # 处理数据
    train_loader, val_loader = train_val_data_process()
    # 开始训练
    train_model(hnet, rnet, train_loader, val_loader)