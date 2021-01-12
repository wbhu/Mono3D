#!/usr/bin/env python
"""
    File Name   :   Mono3D-loss
    date        :   29/11/2019
    Author      :   wenbo
    Email       :   huwenbodut@gmail.com
    Description :
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
"""

import torch.nn.functional as F
import torch
import torch.nn as nn


def grad_l1_loss(y_input, y_target):
    inputGradH = y_input[..., 1:, :] - y_input[..., :-1, :]
    inputGradW = y_input[..., :, 1:] - y_input[..., :, :-1]
    targetGradH = y_target[..., 1:, :] - y_target[..., :-1, :]
    targetGradW = y_target[..., :, 1:] - y_target[..., :, :-1]
    hLoss = F.l1_loss(inputGradH, targetGradH)
    wLoss = F.l1_loss(inputGradW, targetGradW)
    return (hLoss + wLoss) / 2.


class GradSoftL1(nn.Module):
    def __init__(self, ):
        super(GradSoftL1, self).__init__()
        self.l1_fun = CharbonnierLoss()

    def forward(self, y_input, y_target):
        inputGradH = y_input[..., 1:, :] - y_input[..., :-1, :]
        inputGradW = y_input[..., :, 1:] - y_input[..., :, :-1]
        targetGradH = y_target[..., 1:, :] - y_target[..., :-1, :]
        targetGradW = y_target[..., :, 1:] - y_target[..., :, :-1]
        hLoss = self.l1_fun(inputGradH, targetGradH)
        wLoss = self.l1_fun(inputGradW, targetGradW)
        return (hLoss + wLoss) / 2.


class CharbonnierLoss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class MononizingLoss(nn.Module):
    base_loss_f = CharbonnierLoss()

    def __init__(self, cfg=None):
        super(MononizingLoss, self).__init__()
        self.cfg = cfg
        self.grad_loss = GradSoftL1()

    def forward(self, predict, label):
        # base_loss = F.mse_loss(predict, label)
        base_loss = self.base_loss_f(predict, label)
        grad_loss = self.grad_loss(predict, label)
        loss = base_loss + self.cfg.alpha * grad_loss
        return loss


class MononizingLossMSE(MononizingLoss):
    base_loss_f = nn.MSELoss()


class InvertibilityLoss(nn.Module):
    base_loss_f = CharbonnierLoss()

    def __init__(self, cfg=None):
        super(InvertibilityLoss, self).__init__()
        self.cfg = cfg

    def forward(self, predict_l, predict_r, label_l, label_r):
        loss = self.base_loss_f(predict_l, label_l) * (1.0 - self.cfg.right_weight) \
               + self.base_loss_f(predict_r, label_r) * self.cfg.right_weight
        return loss


class InvertibilityLossMSE(InvertibilityLoss):
    base_loss_f = nn.MSELoss()
