# 图像隐写术项目
## 项目介绍
本项目是我学习图像隐写的第一个项目，作为学习和后续再此基础上优化，还需要继续完善，项目实现了基于深度学习的图像隐写术系统，可以将秘密图像隐藏在封面图像中，并能够准确还原出隐藏的图像。项目使用双网络架构(HidingNet + RevealNet)，通过深度学习实现高质量的图像隐写和提取。

## 主要特性
- 双网络架构：使用HidingNet进行图像隐写，RevealNet进行图像提取
- 高质量重建：通过优化网络结构实现高PSNR的图像重建
- 自动化训练：完整的训练和评估流程
- 可视化支持：使用Tensorboard记录训练过程
- 实验管理：自动保存每次实验的模型、图像和日志

## 项目结构
```
├── configs/               # 配置文件
│   ├── __init__.py
│   └── config.py         # 项目配置参数
├── datasets/             # 数据集相关
│   ├── __init__.py
│   ├── data_loader.py    # 数据加载器
│   └── train/           # 训练数据目录
├── models/              # 模型定义
│   ├── __init__.py
│   └── model.py         # 网络模型实现
├── utils/               # 工具函数
│   ├── __init__.py
│   └── metrics.py      # 评价指标（PSNR等）
├── results/            # 训练结果目录
│   └── {device}_{timestamp}/  # 每次实验的结果目录
│       ├── checkpoints/  # 模型检查点
│       ├── images/      # 训练过程图像
│       ├── logs/        # Tensorboard日志
│       └── train.txt    # 训练日志文件
├── requirements.txt    # 项目依赖
├── train.py           # 训练脚本
└── README.md          # 项目文档
```

## 环境配置
1. 创建虚拟环境（推荐）
```bash
conda create -n stegan python=3.9
conda activate stegan
```
2. 安装依赖
```bash
conda create --name stegan --file requirements.txt
```
## 使用方法
1. 准备数据集
   - 将训练图像放入 `datasets/train/` 目录
   - 将验证图像放入 `datasets/val/` 目录

2. 配置参数
   - 在 `configs/config.py` 中修改训练参数
   - 主要参数包括：批次大小、学习率、训练轮数等

3. 开始训练
```bash
python train.py
```

4. 查看结果
   - 训练日志：`results/{device}_{timestamp}/train.txt`
   - 训练过程图像：`results/{device}_{timestamp}/images/`
   - Tensorboard日志：`results/{device}_{timestamp}/logs/`
   - 模型检查点：`results/{device}_{timestamp}/checkpoints/`

## 评估指标
- PSNR (Peak Signal-to-Noise Ratio)：评估重建图像质量
- MSE (Mean Squared Error)：评估图像重建误差

## 注意事项
- 确保有足够的GPU内存
- 建议使用高质量的训练数据
- 可以根据实际需求调整网络结构和训练参数

## 作者
- 作者：xiaonan
- 邮箱：ihshao@outlook.com

## 许可证
本项目采用 MIT 许可证