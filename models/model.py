"""
@Time  : 2025/1/1 上午11:09       
@Author: xiaonan
@Email : itjunhao@qq.com
@File  : model-00.py
"""
import torch
import torch.nn as nn
from torchsummary import summary

# from models.MSFblock import MSFblock

#### 门控融合单元
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class gatedFusion(nn.Module):

    def __init__(self, dim):
        super(gatedFusion, self).__init__()
        self.fc1 = nn.Linear(dim, dim, bias=True)
        self.fc2 = nn.Linear(dim, dim, bias=True)

    def forward(self, x1, x2):
        x11 = self.fc1(x1)
        x22 = self.fc2(x2)
        # 通过门控单元生成权重表示
        z = torch.sigmoid(x11+x22)
        # 对两部分输入执行加权和
        out = z*x1 + (1-z)*x2
        return out



####

class Conv_4x4(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Conv_4x4, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes), # 批量归一化
            nn.LeakyReLU(), # ReLU变体
        )

    def forward(self, x):
        return self.conv(x)

class Deconv_4x4(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Deconv_4x4, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.deconv(x)
class HidingNet(nn.Module):
    def __init__(self):
        super(HidingNet,self).__init__()
        # 下采样层
        self.down = nn.ModuleList([
            Conv_4x4(6, 64),
            Conv_4x4(64, 128),
            Conv_4x4(128, 256),
            Conv_4x4(256, 512),
            Conv_4x4(512, 512),
            Conv_4x4(512, 512),
        ])

        # 中间层
        self.mid = nn.Sequential(
            Conv_4x4(512, 512), # (512, 8, 8)
            Deconv_4x4(512, 512),
        )

        # 上采样层 这里用到了跳跃连接
        self.up = nn.ModuleList([
            Deconv_4x4(512+512, 1024), 
            Deconv_4x4(1024+512, 1024),
            Deconv_4x4(1024+512, 512),
            Deconv_4x4(512+256, 256),
            Deconv_4x4(256+128, 128),
            Deconv_4x4(128+64, 64),
        ])

         #### 门控融合单元
        self.gated_fusions = nn.ModuleList([
            gatedFusion(dim=512),
            gatedFusion(dim=768),
            gatedFusion(dim=768),
            gatedFusion(dim=384),
            gatedFusion(dim=192),
            gatedFusion(dim=96),
        ])

        # 输出层
        self.out = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x  = torch.cat([c, s], dim=1) # 拼接两张照片
        x_list = []
        # 下采样层
        for i in range(6):
            x = self.down[i](x) # 把输入传入每个下采样层
            x_list.append(x)

        # 中间层
        x = self.mid(x)

        # 上采样层
        for i in range(6):
            # x_list[5-i]是对称下采样层的特征，i为0时 5-i为5，表示最后一层下采样层，dim=1表示在通道维度上拼接
            x_cat = torch.cat([x, x_list[5-i]], dim=1)

            #### 分割拼接后的特征
            split_size = int(x_cat.size(1)/2)
            x1, x2 = torch.split(x_cat, [split_size, split_size], dim=1)

            # 修改维度顺序 (B, C, H, W) -> (B, H, W, C)
            x1 = x1.permute(0, 2, 3, 1)
            x2 = x2.permute(0, 2, 3, 1)
            # 应用门控融合单元
            x = self.gated_fusions[i](x1, x2)

            # 修改维度顺序 (B, H, W, C) -> (B, C, H, W)
            x = x.permute(0, 3, 1, 2)
            x = torch.cat([x, x], dim=1) # 拼接跳跃连接的特征
            ###
            x = self.up[i](x)

        # 输出层
        x = self.out(x)
        return x
    
    # 初始化权重
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
# 揭示网络1
class RevealNet(nn.Module):
    def __init__(self, nc=3, nhf=64, output_function=nn.Sigmoid):
        super(RevealNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf * 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            nn.BatchNorm2d(nhf * 4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf * 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(nhf, nc, 3, 1, 1),
            output_function()
        )

    def forward(self, x):
        x = self.main(x)
        return x

# 揭示网络2
class RevealNet_2(nn.Module):
    def __init__(self):
        super(RevealNet_2, self).__init__()
        # 采用和隐藏网络相同的结构
        # 下采样层
        self.down = nn.ModuleList([
            Conv_4x4(3, 64),
            Conv_4x4(64, 128),
            Conv_4x4(128, 256),
            Conv_4x4(256, 512),
            Conv_4x4(512, 512),
            Conv_4x4(512, 512),
        ])

        # 中间层
        self.mid = nn.Sequential(
            Conv_4x4(512, 512),
            Deconv_4x4(512, 512),
        )

        # 上采样层 这里用到了跳跃连接
        self.up = nn.ModuleList([
            Deconv_4x4(512 + 512, 1024),
            Deconv_4x4(1024 + 512, 1024),
            Deconv_4x4(1024 + 512, 512),
            Deconv_4x4(512 + 256, 256),
            Deconv_4x4(256 + 128, 128),
            Deconv_4x4(128 + 64, 64),
        ])

        # 输出层
        self.out = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )
    def forward(self, x):
        # x  = torch.cat([c, s], dim=1) # 拼接两张照片
        x_list = []
        # 下采样层
        for i in range(6):
            x = self.down[i](x) # 把输入传入每个下采样层
            x_list.append(x)

        # 中间层
        x = self.mid(x)

        # 上采样层
        for i in range(6):
            # x_list[5-i]是对称下采样层的特征，i为0时 5-i为5，表示最后一层下采样层，dim=1表示在通道维度上拼接
            x = torch.cat([x, x_list[5-i]], dim=1)
            x = self.up[i](x)

        # 输出层
        x = self.out(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HidingNet().to(device)
    # torch.save(model, 'hnet.pth')
    # print(summary(model, [(3, 256, 256), (3, 256, 256)]))
    print(summary(model, (6, 256, 256)))
