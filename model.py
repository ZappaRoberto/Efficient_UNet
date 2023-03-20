import sys

import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import time

# TODO: Instance segmentation, come realizzarla?
"""
Per l'instance segmentation posso pensare di portarmi con le skip connection features da vari livelli dal decoder
e sfruttare il positional encoding sulla maschera per dare informazione della posizione dei pixel
"""


class DPConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(DPConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels,
                                   out_channels=in_channels,
                                   kernel_size=(3, 3),
                                   padding=1,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=(1, 1),
                                   bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class LKA(nn.Module):
    def __init__(self, channel, bias=False):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=channel,
                      out_channels=channel,
                      kernel_size=(5, 5),
                      padding=2,
                      groups=channel),
            nn.Conv2d(in_channels=channel,
                      out_channels=channel,
                      kernel_size=(7, 7),
                      stride=(1, 1),
                      padding=9,
                      groups=channel,
                      dilation=(3, 3)),
            nn.Conv2d(in_channels=channel,
                      out_channels=channel,
                      kernel_size=(1, 1),
                      bias=bias)
        )

    def forward(self, x):
        return x * self.sequential(x)


class Attention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1)),
            nn.GELU(),
            LKA(channel),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1))
        )

    def forward(self, x):
        return x + self.sequential(x)


class EfficientConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EfficientConvBlock, self).__init__()
        # self.conv1x1 = nn.Conv2d(in_channels, out_channels, (1, 1))
        self.sequential = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Hardswish(),
            DPConv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(),
            DPConv(out_channels, out_channels, True),
        )
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        short = x
        out = self.sequential(x)
        out = self.hardswish(short + out)
        return out


class SPDConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPDConv, self).__init__()
        self.spd = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1=2, s2=2),
            nn.Conv2d(in_channels=in_channels * 4,
                      out_channels=out_channels,
                      kernel_size=(1, 1),
                      bias=False)
        )

    def forward(self, x):
        return self.spd(x)


class FirstDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstDownBlock, self).__init__()
        self.sequential = nn.Sequential(
            DPConv(in_channels, out_channels),
            EfficientConvBlock(out_channels, out_channels),
            Attention(out_channels)
        )

    def forward(self, x):
        return self.sequential(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.sequential = nn.Sequential(
            SPDConv(in_channels, out_channels),
            EfficientConvBlock(out_channels, out_channels),
            Attention(out_channels)
        )

    def forward(self, x):
        return self.sequential(x)


class LinearBottleNeck(nn.Module):
    def __init__(self, out_channels):
        super(LinearBottleNeck, self).__init__()
        self.Conv1x1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (1, 1)),
            # nn.Hardswish(),
            nn.ReLU(),
        )
        self.sequential = nn.Sequential(
            DPConv(out_channels, out_channels, True),
            # nn.Hardswish(),
            nn.ReLU(),
            DPConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (1, 1)),
        )

    def forward(self, x):
        x = self.Conv1x1(x)
        out = self.sequential(x).squeeze()
        return torch.add(x, out)


class BottleNeck(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(BottleNeck, self).__init__()
        self.sequential = nn.Sequential(
            SPDConv(in_channels, middle_channels),
            EfficientConvBlock(middle_channels, middle_channels),
            nn.ConvTranspose2d(in_channels=middle_channels,
                               out_channels=out_channels,
                               kernel_size=(2, 2),
                               stride=(2, 2))
        )

    def forward(self, x):
        return self.sequential(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.convTrans = nn.ConvTranspose2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=(2, 2),
                                            stride=(2, 2))
        self.EfficientConvBlock = EfficientConvBlock(in_channels=in_channels, out_channels=in_channels)
        self.attention = Attention(in_channels)

    def forward(self, x, skip=None):
        # self.doubledsc(self.convTrans(torch.cat((x, self.skip.get()), dim=1)))
        # self.convTrans(self.efficentconvblock(torch.cat((x, self.skip.get()), dim=1)))
        if skip is not None:
            out = torch.cat((x, skip), dim=1)
        else:
            out = x
        out = self.EfficientConvBlock(out)
        out = self.attention(out)
        out = self.convTrans(out)
        return out


class LastLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LastLayer, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        )

    def forward(self, x, skip):
        return self.sequential(torch.cat((x, skip), dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=None):
        super(UNet, self).__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.first = DPConv(in_channels, features[0])  # 3 -> 64
        self.down1 = DownBlock(features[0], features[1])  # 64 -> 128
        self.down2 = DownBlock(features[1], features[2])  # 128 -> 256
        self.down3 = DownBlock(features[2], features[3])  # 256 -> 512
        self.up1 = UpBlock(features[3], features[2])  # 512 -> 256
        self.up2 = UpBlock(features[2] * 2, features[1])  # 256 -> 128
        self.up3 = UpBlock(features[1] * 2, features[0])  # 128 -> 64
        self.last = LastLayer(features[0] * 2, out_channels)  # 64 -> out_channels

    def forward(self, x):
        x0 = self.first(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = self.up1(x3)
        x4 = self.up2(x3, x2)
        x5 = self.up3(x4, x1)
        out = self.last(x5, x0)
        return out


if __name__ == "__main__":
    torch.set_printoptions(edgeitems=32)
    device = torch.device("cuda")
    model = UNet()
    model.eval().to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    sums = 0
    for n in range(60):
        data = torch.rand(1, 3, 256, 256).to(device)
        t0 = time.time()
        m = model(data)
        # print(m.shape)
        t1 = time.time()
        sums += t1 - t0
    print('Mean: {}'.format(sums / 30))
