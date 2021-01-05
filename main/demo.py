#!/usr/bin/env python
"""
    File Name   :   Mono3D-demo
    date        :   13/5/2020
    Author      :   wenbo
    Email       :   huwenbodut@gmail.com
    Description :
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
"""
import cv2
import torch

from base.baseTrainer import state_dict_remove_moudle
from base.utilities import get_logger
from models.p_mbi_pdf import MBIResPDF
from dataset.flickr1024 import Flickr1024, inverse_normalize
import dataset.transform as Xform
from utils import util
from metrics.psnr import PSNR
from metrics.ms_ssim import SSIM


def load_input(left, right, base_resolution=4):
    aug = Xform.Compose([
        Xform.ToTensor(),
        Xform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    left = cv2.cvtColor(cv2.imread(left), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(cv2.imread(right), cv2.COLOR_BGR2RGB)
    H, W, _ = left.shape
    left, origin_shape = Flickr1024.pad_image(left, base_resolution)
    right, origin_shape = Flickr1024.pad_image(right, base_resolution)
    # aug
    left, right = aug(left / 255., right / 255.)
    return left, right, origin_shape


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--left', type=str)
    parser.add_argument('--model', type=str, default='Exp/model_zoo/mono3d_img.pth.tar')
    args = parser.parse_args()
    model_path = args.model

    psnr_cal = PSNR()
    ssim_cal = SSIM(data_range=1.0)
    logger = get_logger()
    model = MBIResPDF(logger, quantize=True)
    model = model.cuda()
    # model.summary(logger, None)
    logger.info("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict_remove_moudle(checkpoint['state_dict']), strict=True)
    # logger.info("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))

    args.right = args.left.replace('_L', '_R')
    left, right, ori_shape = load_input(args.left, args.right)
    name = args.left.split('/')[-1].split('.')[0]


    def fun(x):
        x = Flickr1024.depad_tensor(x, ori_shape)
        x = inverse_normalize(x).clamp(0.0, 1.0)
        return x


    with torch.no_grad():
        model.eval()
        left = left.unsqueeze(0).cuda()
        right = right.unsqueeze(0).cuda()
        res_mono, res_l, res_r = model(left, right)

    left, right, res_mono, res_l, res_r = fun(left), fun(right), fun(res_mono), fun(res_l), fun(res_r)
    psnr = [psnr_cal(res_mono, left), psnr_cal(res_l, left), psnr_cal(res_r, right)]
    ssim = [ssim_cal(res_mono, left), ssim_cal(res_l, left), ssim_cal(res_r, right)]

    print(psnr, '\n', ssim)

    for x, last_fix in zip([res_mono, res_l, res_r], ["_res_mono", "_res_l", "_res_r"]):
        cv2.imwrite(args.left.replace(name, name + last_fix),
                    util.tensor2img(x)[..., ::-1] * 255)
