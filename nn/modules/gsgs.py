import math

import numpy as np
import torch
import torch.nn as nn



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    




    
class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_hw = nn.AdaptiveMaxPool2d((1, 1))
        # self.pool_S = SpatialAttention(3)
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=5, stride=1, padding=2)
        self.conv1 = nn.Conv2d(1, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.branch0 = nn.Sequential(
                #BasicConv(channels, channels//2, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.Conv2d(channels // self.groups // 2, channels // self.groups // 2, kernel_size=3, stride=1, padding=1, dilation=1)
                )
        self.branch1 = nn.Sequential(
                # BasicConv(channels, channels//2, kernel_size=1, stride=1),
                nn.Conv2d(channels // self.groups // 2, channels // self.groups // 2, kernel_size=(3,3), stride=1, padding=(1,1)),#长宽不变
                nn.Conv2d(channels // self.groups // 2, channels // self.groups // 2, kernel_size=3, stride=1, padding=3, dilation=3)
                )
        self.channels=channels
    
    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        
        x_h = self.pool_h(group_x)  # Shape: [b * groups, c // groups, 1, w]
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # Shape: [b * groups, c // groups, h, 1]
        
        ss = self.pool_hw(group_x)  # Shape: [b * groups, c // groups, 1, 1]
        ss = ss.expand(-1, -1, h, w)  # 扩展为 [b * groups, c // groups, h, w]

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # Shape: [b * groups, channels // groups, 1, 1]
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()) #

         # [b * groups, channels // groups, h, w]#3和5的拼接
        part1 = group_x[:, :self.channels // self.groups // 2, :, :]  # 前 3 个通道
        part2 = group_x[:, self.channels // self.groups // 2:, :, :]
        x21 = self.branch0(part1)
        x22 = self.branch1(part2)
        x2  = self.conv3x3(torch.concat((x21,x22),1))
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        #print(f"Input shape: {x.shape}")
        #print(f"Group shape: {group_x.shape}")
        #print(f"Output shape: {((group_x * weights.sigmoid()).reshape(b, c, h, w)).shape}")
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
    
class SpatialAttention(nn.Module):
    """Spatial-attention module."""
 
    def __init__(self, kernel_size=3):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        # self.act = nn.Sigmoid()
 
    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1))
    



class GSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g,1, act)	# g:gract：分组卷积
        self.cv2 = Conv(c_, c_, 5, 1, None, c_,1,act)	# 分组为c_
 
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 1)

class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2,shortcut=True,g=1, k=(3,3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # for lighting
        #self.conv_lighting = nn.Sequential(
            #GSConv(c1, c_, 1, 1),
            #GSConv(c_, c2, 1, 1, act=False))
        # for receptive field
        self.conv = nn.Sequential(	# 没用到
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.add = shortcut and c1 == c2
 
    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.conv(x) if self.add else self.conv(x)
    
class GSC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(GSBottleneck(self.c, self.c, shortcut, g, k=((3,3),(3,3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        
class VoVGSCSP(nn.Module):
    # VoV-GSCSP https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(GSBottleneck(c_, c_) for _ in range(n)))
    def forward(self, x):
        x1 = self.cv1(x)
        return self.cv2(torch.cat((self.m(x1), x1), dim=1))
