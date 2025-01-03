"""
@Time  : 2024/12/5
@Author: xiaonan
@Email : itjunhao@qq.com
@File  : metrics.py
@Description: 图像质量评价指标工具类
"""

import torch
import torch.nn.functional as F

class ImageQualityMetrics:
    """图像质量评价指标类"""
    
    @staticmethod
    def calculate_psnr(original, reconstructed, max_value=1.0):
        """
        计算峰值信噪比(PSNR)
        
        Args:
            original (torch.Tensor): 原始图像
            reconstructed (torch.Tensor): 重建图像 
            max_value (float): 像素最大值
            
        Returns:
            float: PSNR值
        """
        if original.dim() == 3:
            original = original.unsqueeze(0)
            
        mse = F.mse_loss(reconstructed, original)
        psnr = 10 * torch.log10((max_value ** 2) / mse)
        
        return psnr.item() 