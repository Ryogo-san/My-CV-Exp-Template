import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import torch
import torch.nn as nn

from config import CFG
from loss import HeatmapLoss
from models.base_model import BaseModel


class DoubleConv(nn.Module):
    """double conv block

    (conv => [BN] => ReLU) *2
    but with the same image sizes (H,W).

    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding="same"
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=3, stride=1, padding="same"
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    """conv layers for output with pointwise conv and sigmoid"""

    def __init__(self, in_channels):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding="same"),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.out_conv(x)


class UNet(BaseModel):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.double_conv1 = DoubleConv(1, 64)
        self.double_conv2 = DoubleConv(64, 128)
        self.double_conv3 = DoubleConv(128, 256)

        self.double_conv4 = DoubleConv(256, 128)
        self.double_conv5 = DoubleConv(128, 64)
        self.out_conv = OutConv(64)

        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.up_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.criterion = HeatmapLoss()

    def forward(self, x):
        out_1 = self.double_conv1(x)
        out = self.max_pool(out_1)
        out_2 = self.double_conv2(out)
        out = self.max_pool(out_2)
        out_3 = self.double_conv3(out)
        out = self.up_conv1(out_3)
        out = torch.cat((out, out_2), dim=1)
        out_4 = self.double_conv4(out)
        out = self.up_conv2(out_4)
        out = torch.cat((out, out_1), dim=1)
        out_5 = self.double_conv5(out)
        out = self.out_conv(out_5)

        return out


class UNet4(BaseModel):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.double_conv1 = DoubleConv(1, 64)
        self.double_conv2 = DoubleConv(64, 128)
        self.double_conv3 = DoubleConv(128, 256)
        self.double_conv4 = DoubleConv(256, 512)

        self.double_conv5 = DoubleConv(512, 256)
        self.double_conv6 = DoubleConv(256, 128)
        self.double_conv7 = DoubleConv(128, 64)
        self.out_conv = OutConv(64)

        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.up_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.criterion = HeatmapLoss()

    def forward(self, x):
        out_1 = self.double_conv1(x)
        out = self.max_pool(out_1)
        out_2 = self.double_conv2(out)
        out = self.max_pool(out_2)
        out_3 = self.double_conv3(out)
        out = self.max_pool(out_3)
        out_4 = self.double_conv4(out)

        out = self.up_conv1(out_4)
        out = torch.cat((out, out_3), dim=1)
        out_5 = self.double_conv5(out)
        out = self.up_conv2(out_5)
        out = torch.cat((out, out_2), dim=1)
        out_6 = self.double_conv6(out)
        out = self.up_conv3(out_6)
        out = torch.cat((out, out_1), dim=1)
        out_7 = self.double_conv7(out)
        out = self.out_conv(out_7)

        return out


if __name__ == "__main__":
    model = UNet4(CFG)
    print(model)
