#!/usr/bin/env python
"""
    File Name   :   Mono3D-p_mbi_pdf
    date        :   5/5/2020
    Author      :   wenbo
    Email       :   huwenbodut@gmail.com
    Description :
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

from base import BaseModel
from .base_moudle import DeformConvBlock, SoftQuantizer
from models.pyramid_resunet import PyramidResEncoder, PyramidResDecoder


class MBIResPDF(BaseModel):
    LEVELS = 3
    CONV_NUM = 2
    RES_NUM = 2
    DIM = 32

    def __init__(self, logger, quantize=False):
        super().__init__()
        self.logger = logger
        self.quantize = quantize
        self.q = SoftQuantizer()

        self.feature_extractor = PyramidResEncoder(base_dim=self.DIM, levels=self.LEVELS, convNum=self.CONV_NUM)

        self.fusor = PDF()
        self.mono_reconstructor = PyramidResDecoder(base_dim=self.DIM * 2, levels=self.LEVELS, convNum=self.CONV_NUM,
                                                    residual_num=0)
        self.mono_extractor = PyramidResEncoder(base_dim=self.DIM * 2, levels=self.LEVELS, convNum=self.CONV_NUM)
        self.sepor = PDF()
        self.reconstructor = PyramidResDecoder(base_dim=self.DIM, levels=self.LEVELS, convNum=self.CONV_NUM,
                                               residual_num=0)

    def forward(self, left, right):
        # pdb.set_trace()
        N, C, H, W = left.shape
        p_feature = self.feature_extractor(torch.cat([left, right], 0))

        pf = [torch.cat([f[:N], f[N:]], dim=1) for f in p_feature]

        fused_p_feature = self.fusor(*pf)
        res_mono = self.mono_reconstructor(fused_p_feature)
        if self.quantize:
            res_mono = self.q(res_mono)
        mono_f = self.mono_extractor(res_mono)
        p_restore = self.sepor(*mono_f)
        p_l, p_r = [], []
        for p in p_restore:
            _, dim, _, _ = p.shape
            p_l.append(p[:, :dim // 2, ...])
            p_r.append(p[:, dim // 2:, ...])
        res_l = self.reconstructor(p_l)
        res_r = self.reconstructor(p_r)
        return res_mono, res_l, res_r


class PDF(BaseModel):
    PLANES = [32 * 2, 64 * 2, 128 * 2]

    def __init__(self, conv_num=3):
        super(PDF, self).__init__()
        base_block = DeformConvBlock

        self.conv_l3 = base_block(self.PLANES[-1], self.PLANES[-1], convNum=conv_num)
        self.conv_l2 = base_block(self.PLANES[-2], self.PLANES[-2], convNum=conv_num)
        self.conv_l1 = base_block(self.PLANES[-3], self.PLANES[-3], convNum=conv_num)

        self.skip_l2 = nn.Conv2d(self.PLANES[-1] + self.PLANES[-2], self.PLANES[-2], 1, bias=False)
        self.skip_l1 = nn.Conv2d(self.PLANES[-2] + self.PLANES[-3], self.PLANES[-3], 1, bias=False)

    def forward(self, p1, p2, p3):
        pf1 = self.conv_l1(p1)
        pf2 = self.conv_l2(p2)
        pf3 = self.conv_l3(p3)

        pf3_plus = F.interpolate(pf3, size=pf2.shape[-2:], mode='bilinear', align_corners=True)
        pf2 = self.skip_l2(torch.cat([pf2, pf3_plus], dim=1))
        pf2_plus = F.interpolate(pf2, size=pf1.shape[-2:], mode='bilinear', align_corners=True)
        pf1 = self.skip_l1(torch.cat([pf1, pf2_plus], dim=1))

        return pf1, pf2, pf3
