# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act()

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act()
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act()

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)

class PretrainMobileNetv3Small(nn.Module):
    def __init__(self):
        super(PretrainMobileNetv3Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = nn.Hardswish()
        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = nn.Hardswish()

        self.bneck=nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, True, 2),
            Block(3, 16, 72, 24, nn.ReLU, False, 2),
            Block(3, 24, 88, 24, nn.ReLU, False, 1),
            Block(5, 24, 96, 40, nn.Hardswish, True, 2),
            Block(5, 40, 240, 40, nn.Hardswish, True, 1),
            Block(5, 40, 240, 40, nn.Hardswish, True, 1),
            Block(5, 40, 120, 48, nn.Hardswish, True, 1),
            Block(5, 48, 144, 48, nn.Hardswish, True, 1),
            Block(5, 48, 288, 96, nn.Hardswish, True, 2),
            Block(5, 96, 576, 96, nn.Hardswish, True, 1),
            Block(5, 96, 576, 96, nn.Hardswish, True, 1),
        )

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2)

    def forward(self, x, mask):
        mask = self.upsample_mask(mask, 8)
        mask = mask.unsqueeze(1).type_as(x)

        out = self.hs1(self.bn1(self.conv1(x)))

        out = self.bneck[0](out)

        out *= (1.-mask)

        for i in range(len(self.bneck)):
            out = self.bneck[i](out) if i > 0 else out

        out = self.hs2(self.bn2(self.conv2(out)))

        return out

class PretrainMobileNetv3Large(nn.Module):
    def __init__(self):
        super(PretrainMobileNetv3Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = nn.Hardswish()
        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = nn.Hardswish()

        self.bneck=nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, False, 1),
            Block(3, 16, 64, 24, nn.ReLU, False, 2),
            Block(3, 24, 72, 24, nn.ReLU, False, 1),
            Block(5, 24, 72, 40, nn.ReLU, True, 2),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(3, 40, 240, 80, nn.Hardswish, False, 2),
            Block(3, 80, 200, 80, nn.Hardswish, False, 1),
            Block(3, 80, 184, 80, nn.Hardswish, False, 1),
            Block(3, 80, 184, 80, nn.Hardswish, False, 1),
            Block(3, 80, 480, 112, nn.Hardswish, True, 1),
            Block(3, 112, 672, 112, nn.Hardswish, True, 1),
            Block(5, 112, 672, 160, nn.Hardswish, True, 2),
            Block(5, 160, 672, 160, nn.Hardswish, True, 1),
            Block(5, 160, 960, 160, nn.Hardswish, True, 1),
        )

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2)

    def forward(self, x, mask):
        mask = self.upsample_mask(mask, 8)
        mask = mask.unsqueeze(1).type_as(x)

        out = self.hs1(self.bn1(self.conv1(x)))

        out = self.bneck[0](out)
        out = self.bneck[1](out)
        out *= (1.-mask)

        for i in range(len(self.bneck)):
            out = self.bneck[i](out) if i > 1 else out

        out = self.hs2(self.bn2(self.conv2(out)))

        return out