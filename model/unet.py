import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.block(x)


class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.seq = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_c, out_c))

    def forward(self, x): return self.seq(x)


class AttentionGate(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.w_x = nn.Conv2d(ch, ch, 1, bias=False)
        self.w_g = nn.Conv2d(ch, ch, 1, bias=False)
        self.psi = nn.Sequential(nn.Conv2d(ch, 1, 1, bias=False), nn.Sigmoid())

    def forward(self, x, g):
        a = F.relu(self.w_x(x) + self.w_g(g))
        a = self.psi(a)
        return x * a


class Up(nn.Module):
    def __init__(self, in_c, out_c, bilinear=True, use_ag: bool = False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_c // 2, in_c //2, 2, 2)

        self.att: Optional[nn.Module] = AttentionGate(out_c) if use_ag else None
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy, dx = x2.size(2)-x1.size(2), x2.size(3)-x1.size(3)
        x1 = F.pad(x1, [dx//2, dx-dx//2, dy//2, dy-dy//2])

        if self.att is not None:
            x2 = self.att(x2, x1)

        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    def __init__(self, n_channels: int = 3,
                 out_clean: bool = False,
                 use_ag: bool = False):
        super().__init__()
        self.out_clean = out_clean

        self.inc = DoubleConv(n_channels, 64)
        self.d1, self.d2 = Down(64,128), Down(128,256)
        self.d3, self.d4 = Down(256,512), Down(512,512)

        self.u1 = Up(1024,256,use_ag=use_ag)
        self.u2 = Up(512,128,use_ag=use_ag)
        self.u3 = Up(256,64,use_ag=use_ag)
        self.u4 = Up(128,64,use_ag=use_ag)

        self.mask_head = nn.Conv2d(64, 1, 1)
        if self.out_clean:
            self.clean_head = nn.Sequential(nn.Conv2d(64,3,1), nn.Tanh())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)

        x = self.u1(x5,x4)
        x = self.u2(x,x3)
        x = self.u3(x,x2)
        x = self.u4(x,x1)

        mask_logos = self.mask_head(x)
        if self.out_clean:
            clean = (self.clean_head(x)+1)/2
            return mask_logos, clean
        return mask_logos, None
