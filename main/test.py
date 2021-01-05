#!/usr/bin/env python
"""
    File Name   :   Mono3D-test
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
import json

from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from os.path import join
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import cv2

from dataset.flickr1024 import inverse_normalize
from models import get_model
from base.utilities import get_parser, get_logger, worker_init_fn, AverageMeter
from metrics.psnr import PSNR
from metrics.ms_ssim import MS_SSIM, SSIM
from utils import util
from base.baseTrainer import state_dict_remove_moudle

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def main():
    global args, logger
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger = get_logger()
    logger.info(args)

    logger.info("=> creating model ...")
    model = get_model(args, logger)
    model = model.cuda()

    model.summary(logger, None)

    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict_remove_moudle(checkpoint['state_dict']), strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
    else:
        raise RuntimeError("=> no checkpoint flound at '{}'".format(args.model_path))

    model.warm = True

    # ####################### Data Loader ####################### #
    if args.data_name == 'Flickr1024' or args.data_name == 'Extra':
        from dataset.flickr1024 import Flickr1024
        test_data = Flickr1024(data_root=args.data_root, data_list=args.test_set, base_resolution=args.base_resolution,
                               training=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                                  shuffle=False, num_workers=args.test_workers, pin_memory=True,
                                                  drop_last=False,
                                                  worker_init_fn=worker_init_fn)
    else:
        raise Exception('Dataset not supported yet'.format(args.data_name))

    test(model, test_loader, save=True)


def test(model, test_data_loader, save=False):
    # pdb.set_trace()
    psnr_cal = PSNR()
    msssim_cal = MS_SSIM(data_range=1.0)
    ssim_cal = SSIM(data_range=1.0)
    psnr_meter_mono, psnr_meter_resL, psnr_meter_resR = AverageMeter(), AverageMeter(), AverageMeter()
    msssim_meter_mono, msssim_meter_resL, msssim_meter_resR = AverageMeter(), AverageMeter(), AverageMeter()
    ssim_meter_mono, ssim_meter_resL, ssim_meter_resR = AverageMeter(), AverageMeter(), AverageMeter()

    with torch.no_grad():
        model.eval()
        for i, (left, right, original_shape) in enumerate(tqdm(test_data_loader)):
            batch_size = left.shape[0]
            assert batch_size == 1, 'Only support batch of 1 now!'
            left = left.cuda(non_blocking=True)
            right = right.cuda(non_blocking=True)
            res_mono, res_l, res_r = model(left, right)
            original_shape = [x.item() for x in original_shape]

            def fun(x):
                x = test_data_loader.dataset.depad_tensor(x, original_shape)
                x = inverse_normalize(x).clamp(0.0, 1.0)
                return x

            left, right, res_mono, res_l, res_r = fun(left), fun(right), fun(res_mono), fun(res_l), fun(res_r)

            name = test_data_loader.dataset.frames[i][0].split('/')[-1].split('.')[0]

            # pdb.set_trace()
            psnr_meter_mono.update(psnr_cal(res_mono, left), n=batch_size)
            psnr_meter_resL.update(psnr_cal(res_l, left), n=batch_size)
            psnr_meter_resR.update(psnr_cal(res_r, right), n=batch_size)
            msssim_meter_mono.update(msssim_cal(res_mono, left), n=batch_size)
            msssim_meter_resL.update(msssim_cal(res_l, left), n=batch_size)
            msssim_meter_resR.update(msssim_cal(res_r, right), n=batch_size)
            ssim_meter_mono.update(ssim_cal(res_mono, left), n=batch_size)
            ssim_meter_resL.update(ssim_cal(res_l, left), n=batch_size)
            ssim_meter_resR.update(ssim_cal(res_r, right), n=batch_size)

            if save:
                for x, last_fix in zip([res_mono, res_l, res_r], ["_res_mono.png", "_res_l.png", "_res_r.png"]):
                    cv2.imwrite(join(args.save_folder, args.data_name, name + last_fix),
                                util.tensor2img(x)[..., ::-1] * 255)

    logger.info('==>Mononized: \n'
                'PSNR: {psnr_meter_mono.avg:.2f}\n'
                'MS-SSIM: {msssim_meter_mono.avg:.2f}\n'
                'SSIM: {ssim_meter_mono.avg:.2f}\n'
                '==>restored: \n'
                'PSNR: {psnr_meter_resL.avg:.2f}, {psnr_meter_resR.avg:.2f}\n'
                'MS-SSIM: {msssim_meter_resL.avg:.2f}, {msssim_meter_resR.avg:.2f}\n'
                'SSIM: {ssim_meter_resL.avg:.2f}, {ssim_meter_resR.avg:.2f}'.format(
        psnr_meter_mono=psnr_meter_mono, msssim_meter_mono=msssim_meter_mono, ssim_meter_mono=ssim_meter_mono,
        psnr_meter_resL=psnr_meter_resL, msssim_meter_resL=msssim_meter_resL, ssim_meter_resL=ssim_meter_resL,
        psnr_meter_resR=psnr_meter_resR, msssim_meter_resR=msssim_meter_resR, ssim_meter_resR=ssim_meter_resR
    ))


if __name__ == '__main__':
    main()
