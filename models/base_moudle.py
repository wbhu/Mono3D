import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.autograd import Function
from mmdet.ops import ModulatedDeformConvPack

from dataset.flickr1024 import inverse_normalize, normalize


class SoftQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantize = Quantize
        self.inv_nom = inverse_normalize
        self.norm = normalize

    def forward(self, x):
        x = self.inv_nom(x).clamp(0.0, 1.0)
        x = self.quantize.apply(x)
        x = self.norm(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        residual = self.conv(x)
        return x + residual


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, withConvRelu=True):
        super(DownsampleBlock, self).__init__()
        if withConvRelu:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, convNum):
        super(ConvBlock, self).__init__()
        self.inConv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        layers = []
        for _ in range(convNum - 1):
            layers.append(nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.inConv(x)
        x = self.conv(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SkipConnection(nn.Module):
    def __init__(self, channels):
        super(SkipConnection, self).__init__()
        self.conv = nn.Conv2d(2 * channels, channels, 1, bias=False)

    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        return self.conv(x)


class Space2Depth(nn.Module):
    def __init__(self, scaleFactor):
        super(Space2Depth, self).__init__()
        self.scale = scaleFactor
        self.unfold = nn.Unfold(kernel_size=scaleFactor, stride=scaleFactor)

    def forward(self, x):
        (N, C, H, W) = x.size()
        y = self.unfold(x)
        y = y.view((N, int(self.scale * self.scale), int(H / self.scale), int(W / self.scale)))
        return y


class Quantize(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x * 255.
        y = y.round()
        y = y / 255.
        return y

    @staticmethod
    def backward(ctx, grad_output):
        inputX = ctx.saved_tensors
        return grad_output


class QuantizeModule(nn.Module):
    def forward(self, *x):
        return Quantize.apply(x)


class DeformConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, convNum):
        super(DeformConvBlock, self).__init__()
        self.inConv = nn.Sequential(
            ModulatedDeformConvPack(in_channels=inChannels, out_channels=outChannels, kernel_size=3, padding=1,
                                    deformable_groups=2),
            nn.ReLU(inplace=True)
        )
        layers = []
        for _ in range(convNum - 1):
            layers.append(
                ModulatedDeformConvPack(in_channels=inChannels, out_channels=outChannels, kernel_size=3, padding=1,
                                        deformable_groups=2))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.inConv(x)
        x = self.conv(x)
        return x


class Warp(nn.Module):
    def __init__(self, H=256, W=256):
        super(Warp, self).__init__()
        remapW, remapH = np.meshgrid(np.arange(W), np.arange(H))
        reGrid = np.stack((2.0 * remapW / max(W - 1, 1) - 1.0, 2.0 * remapH / max(H - 1, 1) - 1.0), axis=-1)
        reGrid = reGrid[np.newaxis, ...]
        self.grid = torch.from_numpy(reGrid.astype(np.float32))
        normalizer = np.zeros_like(reGrid)
        normalizer[:, :, :, 0] = 2.0 / W
        normalizer[:, :, :, 1] = 2.0 / H
        self.normalizer = torch.from_numpy(normalizer.astype(np.float32))

    def forward(self, x, flow):
        # x is img, in N*C*H*W format
        # flow is in N*2*H*W format
        # flow[:,0,:,:] stand for W direction warp
        grid = self.grid.cuda(flow.get_device()) + self.normalizer.cuda(flow.get_device()) * flow.permute(0, 2, 3, 1)
        return F.grid_sample(x, grid, padding_mode='border', mode='bilinear')
