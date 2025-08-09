#!/usr/bin/env python3
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.seq(x)

class UNetMini(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=16):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.mid = DoubleConv(base*2, base*4)
        self.up1 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.conv1 = DoubleConv(base*4, base*2)
        self.up2 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.conv2 = DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        mid = self.mid(self.pool2(d2))
        u1 = self.up1(mid)
        c1 = self.conv1(torch.cat([u1, d2], dim=1))
        u2 = self.up2(c1)
        c2 = self.conv2(torch.cat([u2, d1], dim=1))
        return self.out(c2)

if __name__ == "__main__":
    net = UNetMini()
    x = torch.randn(1,1,64,64)
    y = net(x)
    print("UNetMini OK:", tuple(y.shape))
