"""
@Time  : 2024/12/5
@Author: xiaonan
@Email : itjunhao@qq.com
@File  : config.py
@Description: 项目配置文件
"""
import os
import time
from datetime import datetime
import socket
import torch

class Config:
    # 数据相关
    Resize = (256, 256) # 图片缩放尺寸
    root_train = "E:\\01data\DIV2K_HR\\train"
    root_val = "datasets/val" 
    batch_size = 2
    num_workers = 2
    
    # 训练相关
    epochs = 100
    learning_rate = 0.001
    train_val_split = 0.8
    
    # 模型相关
    input_channels = 3
    hidden_channels = 64
    
    # 日志相关
    tensorboard_log_dir = "logs"
    checkpoint_dir = "checkpoints"
    results_dir = "results"
    
    # 设备相关
    device = "cuda" if torch.cuda.is_available() else "cpu"  # "cuda" or "cpu"

    # 获取当前时间作为文件夹名
    time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # 设备名+时间作为文件夹名
    run_dir = f"{socket.gethostname()}_{time_str}"
    
    # 结果保存路径
    base_results_dir = "results"
    current_results_dir = os.path.join(base_results_dir, run_dir)
    
    # 创建子目录
    checkpoints_dir = os.path.join(current_results_dir, "checkpoints")
    images_dir = os.path.join(current_results_dir, "images")
    logs_dir = os.path.join(current_results_dir, "logs")
    # 添加训练日志文件路径
    train_log_file = os.path.join(current_results_dir, "train.txt")
