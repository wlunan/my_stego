"""
@Time  : 2024/12/7 10:30       
@Author: xiaonan
@Email : itjunhao@qq.com
@File  : data_loader.py        
"""
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class BatchSplitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
            Args:
                data_dir (str): 数据集所在目录
                transform (callable, optional): 图像变换函数
        """
        self.root_dir = root_dir
        self.transform = transform
        #  将筛选出的文件名与目录路径组合成完整的文件路径。
        self.file_paths = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 获取文件路径
        file_path = self.file_paths[idx]

        # 打开图像
        image = Image.open(file_path).convert('RGB')  # 转为 RGB 图像

        # 应用变换
        if self.transform:
            image = self.transform(image)
            type(image)
        # return image, file_path  # 返回图像和文件路径
        return image # 返回图像


