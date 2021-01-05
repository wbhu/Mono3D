#!/usr/bin/env python
"""
    File Name   :   Mono3D-pyramid_unet
    date        :   3/4/2020
    Author      :   wenbo
    Email       :   huwenbodut@gmail.com
    Description :
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
"""

from base.base_model import BaseModel
import torch.nn as nn
from .base_moudle import ConvBlock, DownsampleBlock, ResidualBlock, SkipConnection, UpsampleBlock


class PyramidResEncoder(BaseModel):
    def __init__(self, base_dim=32, levels=3, convNum=2, in_channels=3):
        super(PyramidResEncoder, self).__init__()
        self.in_planes = base_dim
        self.levels = levels

        self.in_conv = ConvBlock(in_channels, base_dim, convNum=1)

        self.feature_extractor = []
        for L in range(self.levels):
            # pdb.set_trace()
            if L == 0:
                self.feature_extractor.append(
                    nn.Sequential(*[ResidualBlock(base_dim) for _ in range(convNum)])
                )
            else:
                self.feature_extractor.append(
                    nn.Sequential(
                        DownsampleBlock(base_dim * (2 ** (L - 1)), base_dim * (2 ** L), withConvRelu=False),
                        *[ResidualBlock(base_dim * (2 ** L)) for _ in range(convNum)]
                    )
                )
        self.feature_extractor = nn.ModuleList(self.feature_extractor)

    def forward(self, x):
        x = self.in_conv(x)
        features = []
        for f in self.feature_extractor:
            x = f(x)
            features.append(x)
        return features


class PyramidResDecoder(BaseModel):
    def __init__(self, base_dim=32, levels=3, convNum=2, residual_num=2, out_channels=3):
        super(PyramidResDecoder, self).__init__()
        self.in_planes = base_dim
        self.levels = levels

        self.residual = nn.Sequential(
            *[ResidualBlock(base_dim * (2 ** (self.levels - 1))) for _ in range(residual_num)])

        self.uper, self.skipper = [], []
        for L in list(range(self.levels))[::-1][:-1]:
            self.uper.append(
                nn.Sequential(
                    UpsampleBlock(base_dim * (2 ** L), base_dim * (2 ** (L - 1))),
                    *[ResidualBlock(base_dim * (2 ** (L - 1))) for _ in range(convNum)]
                )
            )
            self.skipper.append(SkipConnection(base_dim * (2 ** (L - 1))))
        self.uper = nn.ModuleList(self.uper)
        self.skipper = nn.ModuleList(self.skipper)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, out_channels, kernel_size=1, padding=0)
            # , nn.Tanh()
        )

    def forward(self, pyramid_features):
        x = self.residual(pyramid_features[-1])
        for i in range(self.levels - 1):
            x = self.uper[i](x)
            x = self.skipper[i](x, pyramid_features[-i - 2])
        x = self.out_conv(x)
        return x

