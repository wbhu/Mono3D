from models.base_moudle import Warp
from torch import nn
import torch
import torch.nn.functional as F
from .jpg_module import JPGQuantizeFun


class Jitter(nn.Module):
    def __init__(self, H=256, W=256, blockSize=16, prob=1):
        super(Jitter, self).__init__()
        self.warper = Warp(H, W)
        self.blockSize = blockSize
        self.prob = prob
        self.H = H
        self.W = W

    def forward(self, x):
        # x is img, in N*C*H*W format
        batchSize = x.shape[0]
        randomFlow = torch.rand(size=(batchSize, 2, self.H // self.blockSize, self.W // self.blockSize),
                                device=x.get_device()) * 2. - 1.
        randomFlow *= self.prob
        randomFlow = F.interpolate(randomFlow, (self.H, self.W), mode='nearest')
        return self.warper(x, randomFlow)


class InterFrameNoiser(nn.Module):
    def __init__(self, H=256, W=256, blockSize=16, prob=1, jpegQuality=0.7):
        super(InterFrameNoiser, self).__init__()
        self.jitter = Jitter(H, W, blockSize, prob)
        self.jpeg = JPGQuantizeFun(jpg_quality=jpegQuality)

    def forward(self, x):
        noised = self.jitter(x)
        residual = x - noised
        residualJpeg = self.jpeg(residual)
        return noised + residualJpeg
