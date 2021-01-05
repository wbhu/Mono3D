#!/usr/bin/env python
"""
    File Name   :   LLTorch-psnr
    date        :   4/6/2019
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
import torch.nn as nn


class PSNR(nn.Module):
    def __init__(self, transform=None):
        super(PSNR, self).__init__()
        self.transform = transform

    def forward(self, img1, img2):
        """
        :param img1: N*C*H*W cuda tensor
        :param img2: N*C*H*W cuda tensor
        :return: psnr
        """
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        mse = (img1 - img2) ** 2
        mse = mse.mean(dim=(1, 2, 3))
        psnr = -10.0 * torch.log10(mse)
        return psnr.mean()
