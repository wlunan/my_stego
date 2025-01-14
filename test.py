"""
@Time  : 2024/12/5
@Author: xiaonan
@Email : itjunhao@qq.com
@File  : test.py
@Description: 模型测试脚本
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from models.model import HidingNet, RevealNet
from utils.metrics import ImageQualityMetrics
from configs.config import Config as cfg

def load_models(hnet_path, rnet_path):
    """
    加载训练好的模型
    
    Args:
        hnet_path: HidingNet模型路径
        rnet_path: RevealNet模型路径
    """
    device = torch.device(cfg.device)
    
    hnet = HidingNet().to(device)
    rnet = RevealNet().to(device)
    
    hnet.load_state_dict(torch.load(hnet_path))
    rnet.load_state_dict(torch.load(rnet_path))
    
    hnet.eval()
    rnet.eval()
    
    return hnet, rnet

def load_and_process_image(image_path):
    """
    加载并处理图像
    
    Args:
        image_path: 图像路径
    """
    transform = transforms.Compose([
        transforms.Resize(cfg.Resize),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path)
    image = transform(image)
    # return image.unsqueeze(0)  # 添加batch维度
    return image

def save_test_results(cover_img, secret_img, combined_img, decoded_img, save_dir):
    """
    保存测试结果
    
    Args:
        cover_img: 封面图像
        secret_img: 秘密图像
        combined_img: 合成图像
        decoded_img: 解码图像
        save_dir: 保存目录
    """
    plt.figure(figsize=(20, 10))
    
    # 显示原始封面图像
    plt.subplot(2, 2, 1)
    plt.title('Cover Image')
    plt.imshow(cover_img.squeeze().permute(1, 2, 0).cpu())
    plt.axis('off')
    
    # 显示原始秘密图像
    plt.subplot(2, 2, 2)
    plt.title('Secret Image')
    plt.imshow(secret_img.squeeze().permute(1, 2, 0).cpu())
    plt.axis('off')
    
    # 显示合成图像
    plt.subplot(2, 2, 3)
    plt.title('Combined Image')
    plt.imshow(combined_img.squeeze().permute(1, 2, 0).detach().cpu())
    plt.axis('off')
    
    # 显示解码后的秘密图像
    plt.subplot(2, 2, 4)
    plt.title('Decoded Secret Image')
    plt.imshow(decoded_img.squeeze().permute(1, 2, 0).detach().cpu())
    plt.axis('off')
    
    # 保存结果
    plt.savefig(os.path.join(save_dir, 'test_results.png'))
    plt.close()

def test_model(hnet, rnet, cover_img, secret_img):
    """
    测试模型
    
    Args:
        hnet: HidingNet模型
        rnet: RevealNet模型
        cover_img: 封面图像
        secret_img: 秘密图像
    """
    device = torch.device(cfg.device)
    
    # 将图像移到设备上
    cover_img = cover_img.to(device)
    secret_img = secret_img.to(device)
    
    # 合并图像
    combined = torch.cat((cover_img, secret_img), dim=0)
    combined = combined.unsqueeze(0)
    
    with torch.no_grad():
        # 生成合成图像
        combined_img = hnet(combined)
        # 解码秘密图像
        decoded_img = rnet(combined_img)
        
        # 计算PSNR
        cover_psnr = ImageQualityMetrics.calculate_psnr(cover_img, combined_img)
        secret_psnr = ImageQualityMetrics.calculate_psnr(secret_img, decoded_img)
        
    return combined_img, decoded_img, cover_psnr, secret_psnr

def main():
    # 创建测试结果目录
    test_results_dir = os.path.join(cfg.base_results_dir, 'test_results')
    os.makedirs(test_results_dir, exist_ok=True)
    
    # 加载模型
    hnet_path = 'results/epoch_291_HLoss_0.0012_RLoss_0.0013_PSNR_29.35_hnet.pth'  # 替换为实际的模型路径
    rnet_path = 'results/epoch_291_HLoss_0.0012_RLoss_0.0013_PSNR_29.35_rnet.pth'  # 替换为实际的模型路径
    hnet, rnet = load_models(hnet_path, rnet_path)
    
    # 加载测试图像
    cover_img_path = 'datasets/0890.png'  # 替换为实际的图像路径
    secret_img_path = 'datasets/0898.png'  # 替换为实际的图像路径
    
    cover_img = load_and_process_image(cover_img_path)
    secret_img = load_and_process_image(secret_img_path)
    
    # 测试模型
    combined_img, decoded_img, cover_psnr, secret_psnr = test_model(hnet, rnet, cover_img, secret_img)
    
    # 保存结果
    save_test_results(cover_img, secret_img, combined_img, decoded_img, test_results_dir)
    
    # 保存载密图像和解密图像
    # 保存载密图像
    vutils.save_image(
        combined_img,
        os.path.join(test_results_dir, 'combined_image.png'),
        normalize=True
    )
    
    # 保存解密图像
    vutils.save_image(
        decoded_img,
        os.path.join(test_results_dir, 'decoded_image.png'),
        normalize=True
    )
    
    # 打印评估结果
    print(f'Cover Image PSNR: {cover_psnr:.2f} dB')
    print(f'Secret Image PSNR: {secret_psnr:.2f} dB')

if __name__ == '__main__':
    main()
