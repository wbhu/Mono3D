#!/usr/bin/env python
"""
    File Name   :   LLTorch-ssim
    date        :   30/5/2019
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
import numpy as np
import math


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Returns a 2D Gaussian kernel array."""
    tmp = np.arange(size).astype(np.float32) - size // 2
    kern1d = (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-0.5 * (tmp / sigma) ** 2)
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


class SSIM(torch.nn.Module):
    # The results are a little different from skimage.measure.compare_ssim for efficient concern,
    # for accurate evaluation please use other credible libraries.
    def __init__(self, channel=3, windowSize=11, reduceMean=True, transform=None):
        super(SSIM, self).__init__()
        self.transform = transform
        self.windowSize = windowSize
        self.reduceMean = reduceMean
        self.channel = channel
        self.kernel = gaussian_kernel(windowSize, sigma=1.5)[np.newaxis, np.newaxis, :, :].astype(np.float32)
        if channel > 1:
            self.kernel = np.concatenate([self.kernel for _ in range(channel)], axis=0)
        self.kernel = torch.from_numpy(self.kernel)

    def forward(self, img1, img2):
        """
        :param img1: N*C*H*W cuda tensor
        :param img2: N*C*H*W cuda tensor
        :return: ssim or ssim map
        """
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        device = img1.get_device()
        mu1 = F.conv2d(img1, self.kernel.cuda(device), padding=self.windowSize // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.kernel.cuda(device), padding=self.windowSize // 2, groups=self.channel)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.kernel.cuda(device), padding=self.windowSize // 2,
                             groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.kernel.cuda(device), padding=self.windowSize // 2,
                             groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.kernel.cuda(device), padding=self.windowSize // 2,
                           groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.reduceMean:
            return ssim_map.mean()
        else:
            return ssim_map.mean(dim=1, keepdim=True)
