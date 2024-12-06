"""
@Time  : 2024/12/5 21:16       
@Author: xiaonan
@Email : itjunhao@qq.com
@File  : model.py        
"""

import torch
import torch.nn as nn
from torchsummary import summary
class HidingNet(nn.Module):
    def __init__(self):
        super(HidingNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class RevealNet(nn.Module):
    def __init__(self):
        super(RevealNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HidingNet = HidingNet().to(device)
    RevealNet = RevealNet().to(device)
    print(summary(HidingNet, (6, 256, 256)))
    print(summary(RevealNet, (64, 32, 32)))