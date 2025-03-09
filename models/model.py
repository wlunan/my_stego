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

########

import math
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

"""SHISRCNet: Super-resolution And Classification Network For Low-resolution Breast Cancer Histopathology Image"""

class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_sizes, padding = paddings, dilation = dilations, bias=False),###, bias=False
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class MSFblock(nn.Module):
    def __init__(self, in_channels):
        super(MSFblock, self).__init__()
        out_channels = in_channels

        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)
            #nn.Dropout(0.5))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim = 2)
        self.Sigmoid = nn.Sigmoid()
        self.SE1 = oneConv(in_channels,in_channels,1,0,1)
        self.SE2 = oneConv(in_channels,in_channels,1,0,1)
        self.SE3 = oneConv(in_channels,in_channels,1,0,1)
        self.SE4 = oneConv(in_channels,in_channels,1,0,1)

    def forward(self, x0,x1,x2,x3):
        # x1/x2/x3/x4: (B,C,H,W)
        y0 = x0
        y1 = x1
        y2 = x2
        y3 = x3

        # 通过池化聚合全局信息,然后通过1×1conv建模通道相关性: (B,C,H,W)-->GAP-->(B,C,1,1)-->SE1-->(B,C,1,1)
        y0_weight = self.SE1(self.gap(x0))
        y1_weight = self.SE2(self.gap(x1))
        y2_weight = self.SE3(self.gap(x2))
        y3_weight = self.SE4(self.gap(x3))

        # 将多个尺度的全局信息进行拼接: (B,C,4,1)
        weight = torch.cat([y0_weight,y1_weight,y2_weight,y3_weight],2)
        # 首先通过sigmoid函数获得通道描述符表示, 然后通过softmax函数,求每个尺度的权重: (B,C,4,1)--> (B,C,4,1)
        weight = self.softmax(self.Sigmoid(weight))

        # weight[:,:,0]:(B,C,1); (B,C,1)-->unsqueeze-->(B,C,1,1)
        y0_weight = torch.unsqueeze(weight[:,:,0],2)
        y1_weight = torch.unsqueeze(weight[:,:,1],2)
        y2_weight = torch.unsqueeze(weight[:,:,2],2)
        y3_weight = torch.unsqueeze(weight[:,:,3],2)

        # 将权重与对应的输入进行逐元素乘法: (B,C,1,1) * (B,C,H,W)= (B,C,H,W), 然后将多个尺度的输出进行相加
        x_att = y0_weight*y0+y1_weight*y1+y2_weight*y2+y3_weight*y3
        return self.project(x_att)


# if __name__ == '__main__':
#     # (B,C,H,W)
#     x0 = torch.rand(1, 64, 192, 192)
#     x1 = torch.rand(1, 64, 192, 192)
#     x2 = torch.rand(1, 64, 192, 192)
#     x3 = torch.rand(1, 64, 192, 192)
#     Model = MSFblock(in_channels=64)
#     out = Model(x0,x1,x2,x3)
#     print(out.shape)

#########

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

        # MSFBlock
        self.msf_block = MSFblock(512)  # 512是 `mid` 层的通道数

        # 上采样层 这里用到了跳跃连接
        self.up = nn.ModuleList([
            Deconv_4x4(512+512, 1024),
            Deconv_4x4(1024+512, 1024),
            Deconv_4x4(1024+512, 512),
            Deconv_4x4(512+256, 256),
            Deconv_4x4(256+128, 128),
            Deconv_4x4(128+64, 64),
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

        ### 用MSF获取多个尺度的特征
        # 通过 MSFblock 处理
        x = self.msf_block(x, x, x, x)

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
    print(summary(model, [(3, 256, 256), (3, 256, 256)]))
