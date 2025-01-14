"""
@Time  : 2024/12/5 21:53       
@Author: xiaonan
@Email : itjunhao@qq.com
@File  : train.py        
"""
import copy
import time
import os
from datetime import datetime

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
from configs.config import Config as cfg
from utils.metrics import ImageQualityMetrics  # 导入评价指标工具类


def train_val_data_process():
    ROOT_TRAIN = cfg.root_train
    ROOT_VAL = cfg.root_val
    BATCH_SIZE = cfg.batch_size

    train_transform = transforms.Compose([
      transforms.Resize(cfg.Resize),transforms.ToTensor()
    ])
    # 加载数据集
    dataset = BatchSplitDataset(root_dir=ROOT_TRAIN, transform=train_transform)
    # 划分训练集和验证集
    train_data, val_data = data.random_split(dataset,
                                             lengths=[round(0.8*len(dataset)), round(0.2*len(dataset))])

    train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=cfg.num_workers)
    val_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=cfg.num_workers)

    return train_loader, val_loader

def validate_model(hnet, rnet, val_loader, device):
    """
    在验证集上评估模型
    
    Args:
        hnet: HidingNet模型
        rnet: RevealNet模型
        val_loader: 验证数据加载器
        device: 设备
    """
    hnet.eval()
    rnet.eval()
    
    total_cover_psnr = 0
    total_secret_psnr = 0
    val_count = 0
    
    with torch.no_grad():
        for cover_img, secret_img in val_loader:
            # 将数据移到设备上
            cover_img = cover_img.to(device)
            secret_img = secret_img.to(device)
            
            # 合并图像
            combined = torch.cat((cover_img, secret_img), dim=0)
            combined = combined.unsqueeze(0)
            
            # 前向传播
            combined_img = hnet(combined)
            decoded_img = rnet(combined_img)
            
            # 计算PSNR
            cover_psnr = ImageQualityMetrics.calculate_psnr(cover_img, combined_img)
            secret_psnr = ImageQualityMetrics.calculate_psnr(secret_img, decoded_img)
            # 计算mse
            # cover_mse = F.mse_loss(cover_img, combined_img)
            # secret_mse = F.mse_loss(secret_img, decoded_img)

            total_cover_psnr += cover_psnr
            total_secret_psnr += secret_psnr
            val_count += 1
    
    # 计算平均PSNR
    avg_cover_psnr = total_cover_psnr / val_count
    avg_secret_psnr = total_secret_psnr / val_count
    
    return avg_cover_psnr, avg_secret_psnr

def train_model(hnet, rnet, train_loader, val_loader, writer):
    # # 创建本次训练的所有相关目录
    # cfg.create_dirs()
    
    device = torch.device(cfg.device)
    # 修改tensorboard日志保存路径
    # writer = SummaryWriter(cfg.logs_dir)
    
    # 最佳权重
    best_hnet_model = copy.deepcopy(hnet.state_dict)
    best_rnet_model = copy.deepcopy(rnet.state_dict)

    # 定义损失函数和优化器
    mse_fun = nn.MSELoss()

    r_optimizer = torch.optim.Adam(hnet.parameters(), lr=0.001)
    h_optimizer = torch.optim.Adam(rnet.parameters(), lr=0.001)

    hnet.to(device)
    rnet.to(device)

    # 训练前设置的参数
    epochs = cfg.epochs
    start_time = time.time()
    min_mseloss = 1 # 每个图像的最小损失

    # 开始训练
    hnet.train()
    rnet.train()

    for epoch in range(epochs):
        # 每个批次设置的参数
        last_cover_img = None
        last_combined_img = None
        last_secret_img = None
        last_rev_secret_img = None

        total_hnet_mseloss = 0.0  # 隐藏网络的损失 封面图像和结合图像之间的损失
        cover_nums = 0
        total_rnet_mseloss = 0.0  # 秘密图像的损失 秘密图像和解密图像之间的损失
        secret_nums = 0
        total_rnet_psnr = 0.0  # 一个epoch中的揭示的总的PSNR
        total_mseloss = 0.0  # 总的损失

        for step, (cover_img, secret_img) in enumerate(train_loader):
            # print(step, cover_img.shape, secret_img.shape)
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
            total_mseloss += total_loss.item()

            # 使用评价指标工具类计算PSNR
            psnr_value = ImageQualityMetrics.calculate_psnr(secret_img, rev_secret_img)
            total_rnet_psnr += psnr_value

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
        ### 每个epoch结束后的操作
        # 计算每个epoch的平均损失
        avg_hnet_mseloss = total_hnet_mseloss / cover_nums
        avg_rnet_mseloss = total_rnet_mseloss / secret_nums
        avg_psnr = total_rnet_psnr / secret_nums if secret_nums > 0 else 0
        avg_total_mseloss = avg_hnet_mseloss + avg_rnet_mseloss
        # 保存模型 当前epoch次的总损失小于最小总损失,并且epoch是10的倍数,减少保存次数
        # if avg_total_mseloss < min_mseloss and epoch % 10 == 0:
        #     min_mseloss = avg_total_mseloss
        #     best_hnet_model = copy.deepcopy(hnet.state_dict())
        #     best_rnet_model = copy.deepcopy(rnet.state_dict())
        #
        #     # 使用新的保存路径
        #     checkpoint_name = f'epoch_{epoch + 1}_HLoss_{avg_hnet_mseloss:.4f}_RLoss_{avg_rnet_mseloss:.4f}_PSNR_{avg_psnr:.2f}'
        #     hnet_path = os.path.join(cfg.checkpoints_dir, f'{checkpoint_name}_hnet.pth')
        #     rnet_path = os.path.join(cfg.checkpoints_dir, f'{checkpoint_name}_rnet.pth')
        #
        #     torch.save(best_hnet_model, hnet_path)
        #     torch.save(best_rnet_model, rnet_path)

        # 记录训练损失
        writer.add_scalar("Hnet Loss", avg_hnet_mseloss, epoch+1)
        writer.add_scalar("Rnet Loss", avg_rnet_mseloss, epoch+1)
        writer.add_scalar("secrect psnr Loss", avg_psnr, epoch+1)

        end_time = time.time()

        # 输出当前 epoch 的训练损失和 PSNR
        # print(f'Epoch [{epoch + 1}/{epochs}], HLoss: {total_hnet_mseloss / secret_nums:.4f}, PSNR: {avg_psnr:.2f} dB, Time: {end_time - start_time:.2f} s')
        # 记录的损失是每个epoch的平均损失
        log_print(
            f'Epoch [{epoch + 1}/{epochs}], '
            f'HLoss: {avg_hnet_mseloss:.4f}, '
            f'RLoss: {avg_rnet_mseloss:.4f}, '
            f'PSNR: {avg_psnr:.2f}dB, '
            f'Time: {end_time - start_time:.2f}s'
        )
        # print(f'Epoch [{epoch + 1}/{epochs}], Loss: {sum_mseloss / secret_nums:.4f} Time: {end_time - start_time:.2f} s')

        # 可视化最后一个批次的四个图像
        visualize_images(last_cover_img, last_secret_img, last_combined_img, last_rev_secret_img, epoch)

        # 验证阶段
        val_cover_psnr, val_secret_psnr = validate_model(hnet, rnet, val_loader, device)
        best_val_psnr = 0
        total_val_psnr = val_cover_psnr+val_secret_psnr
        # 记录验证结果
        writer.add_scalar("Val/Cover PSNR", val_cover_psnr, epoch+1)
        writer.add_scalar("Val/Secret PSNR", val_secret_psnr, epoch+1)
        
        # 打印训练和验证结果
        log_print(
            f'Epoch [{epoch + 1}/{epochs}], '
            f'Train - HLoss: {avg_hnet_mseloss:.4f}, '
            f'RLoss: {avg_rnet_mseloss:.4f}, '
            f'Sectet PSNR: {avg_psnr:.2f}dB, '
            f'Val - Cover PSNR: {val_cover_psnr:.2f}dB, '
            f'Secret PSNR: {val_secret_psnr:.2f}dB, '
            f'Time: {end_time - start_time:.2f}s'
        )
        
        # 根据验证集性能保存最佳模型
        if total_val_psnr > best_val_psnr and epoch % 10 == 0:
            best_val_psnr = total_val_psnr
            checkpoint_name = f'epoch_{epoch + 1}_HLoss_{avg_hnet_mseloss:.4f}_RLoss_{avg_rnet_mseloss:.4f}_PSNR_{val_secret_psnr:.2f}'
            hnet_path = os.path.join(cfg.checkpoints_dir, f'{checkpoint_name}_hnet.pth')
            rnet_path = os.path.join(cfg.checkpoints_dir, f'{checkpoint_name}_rnet.pth')
            
            torch.save(hnet.state_dict(), hnet_path)
            torch.save(rnet.state_dict(), rnet_path)
            log_print(f'保存最佳模型 - Val total PSNR: {total_val_psnr:.2f}dB')

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

    # 修改图像保存路径
    save_path = os.path.join(cfg.images_dir, f'epoch_{epoch + 1}_images.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 添加这行来关闭图像，防止内存泄漏

def create_dirs():
    # 创建必要的目录
    os.makedirs(cfg.current_results_dir, exist_ok=True)
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    os.makedirs(cfg.images_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)
    
    # 创建并初始化日志文件
    with open(cfg.train_log_file, 'w', encoding='utf-8') as f:
        f.write(f"训练开始时间: {cfg.time_str}\n")
        f.write("-" * 50 + "\n")

def log_print(*args, **kwargs):
    """
    同时将信息打印到控制台和写入到日志文件
    """
    print(*args, **kwargs)  # 打印到控制台
    # 将信息写入到日志文件
    with open(cfg.train_log_file, 'a', encoding='utf-8') as f:
        print(*args, file=f, **kwargs)  # 写入到文件

if __name__ == '__main__':

    create_dirs()
    # 创建writer
    writer = SummaryWriter(cfg.logs_dir)
    
    log_print(f"开始训练 - 保存路径: {cfg.current_results_dir}")
    hnet = model.HidingNet()
    rnet = model.RevealNet_2()
    
    device = torch.device(cfg.device)
    hnet = hnet.to(device)
    rnet = rnet.to(device)
    
    train_loader, val_loader = train_val_data_process()
    log_print('------------------开始训练------------------')
    
    train_model(hnet, rnet, train_loader, val_loader, writer)
    writer.close()
    