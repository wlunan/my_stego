"""
@Time  : 2024/12/7 19:15       
@Author: xiaonan
@Email : itjunhao@qq.com
@File  : test.py        
"""
from configs.config import Config
import datasets.data_loader
from torch.utils import data
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# 这是测试数据导入的代码
def test1():
    root_train = Config.root_train
    trans = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor()])
    # 加载数据集
    dataset = datasets.data_loader.BatchSplitDataset(root_dir=root_train, transform=trans)
    rain_loader = data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    print(len(dataset) ,len(rain_loader)) # 前者是返回的数据集的长度，后者是返回的是根据batchsize为2的批次数
    for i, image in enumerate(rain_loader):
        print(i, image.size())

        # plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(image[0].permute(1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(image[1].permute(1, 2, 0))
        plt.show()
        break

# 计算两幅图像的 PSNR、SSIM
def test2():
    from skimage.metrics import structural_similarity as ssim
    import torch
    import torchvision.transforms as transform
    import PIL.Image as Image
    import numpy as np

    # 计算两幅图像的 PSNR（Peak Signal-to-Noise Ratio）
    def calculate_psnr(img1, img2):
        """
        计算两幅图像的 PSNR（Peak Signal-to-Noise Ratio）

        :param img1: 原始图像，形状为 (C, H, W) 或 (N, C, H, W)
        :param img2: 处理后的图像，形状为 (C, H, W) 或 (N, C, H, W)
        :return: PSNR 值（以 dB 为单位）
        """
        # 确保两幅图像形状相同
        assert img1.shape == img2.shape, "Input images must have the same dimensions."

        # 计算 MSE
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')  # 如果 MSE 为零，PSNR 无穷大

        # 假设图像像素范围为 [0, 1]
        max_pixel = 1
        psnr = 10 * torch.log10((max_pixel ** 2) / mse)
        return psnr.item()

    # 加载图像并进行预处理
    def load_image(image_path, transform=None):
        """
        加载图像并进行预处理
        :param image_path: 图像文件路径
        :param transform: 可选的预处理操作
        :return: 处理后的图像 tensor
        """
        image = Image.open(image_path).convert('RGB')  # 确保是 RGB 图像
        if transform:
            image = transform(image)
        return image

    # 计算 SSIM（结构相似性指数）
    def calculate_ssim(img1, img2):
        """
        计算两幅图像的 SSIM（结构相似性指数）
        :param img1: 图像1
        :param img2: 图像2
        :return: SSIM 值
        """
        # 将图像从 tensor 转换为 NumPy 数组，并转为灰度图
        img1 = img1.squeeze().cpu().numpy().transpose(1, 2, 0)  # 转换为 HxWxC
        img2 = img2.squeeze().cpu().numpy().transpose(1, 2, 0)  # 转换为 HxWxC
        img1_gray = np.mean(img1, axis=-1)  # 将 RGB 转为灰度图
        img2_gray = np.mean(img2, axis=-1)  # 将 RGB 转为灰度图

        # 计算 SSIM
        ssim_index, _ = ssim(img1_gray, img2_gray, full=True, data_range=1)
        return ssim_index

    # 示例用法
    if __name__ == "__main__":
        transforms = transform.Compose([
            transform.ToTensor(),
            transform.Resize((256, 256))
        ])

        # 导入图像
        img1_path = '../datasets/0002.png'
        img2_path = '../datasets/0002-噪声10%.png'
        img1 = load_image(img1_path, transforms)
        img2 = load_image(img2_path, transforms)

        # 计算 PSNR
        psnr_value = calculate_psnr(img1, img2)
        print(f"PSNR: {psnr_value:.2f} dB")

        # 计算 SSIM
        ssim_value = calculate_ssim(img1, img2)
        print(f"SSIM: {ssim_value:.4f}")


if __name__ == '__main__':
    test1()
    # test2()