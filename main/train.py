#!/usr/bin/env python
"""
    File Name   :   Mono3D-train
    date        :   3/11/2019
    Author      :   wenbo
    Email       :   huwenbodut@gmail.com
    Description :
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
"""

import os
import time
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
import cv2
import pdb

from base.baseTrainer import poly_learning_rate, reduce_tensor, \
    save_checkpoint_roboust, load_state_dict_roboust
from base.utilities import get_parser, get_logger, main_process, worker_init_fn, AverageMeter
from dataset.flickr1024 import inverse_normalize
from models import get_model
from metrics.loss import *
from metrics import psnr, ssim

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        # cudnn.benchmark = False
        # cudnn.deterministic = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    # global args
    # global best_metric
    cfg = args
    best_metric = 1e10

    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url, world_size=cfg.world_size,
                                rank=cfg.rank)

    # ####################### Model ####################### #
    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(cfg.save_path)

    model = get_model(cfg, logger)
    if cfg.sync_bn:
        logger.info("using DDP synced BN")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if main_process(cfg):
        logger.info(cfg)
        logger.info("=> creating model ...")
        model.summary(logger, writer)

    if cfg.distributed:
        torch.cuda.set_device(gpu)
        cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
        cfg.batch_size_val = int(cfg.batch_size_val / ngpus_per_node)
        cfg.workers = int(cfg.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=True)
    else:
        # model = model.cuda()
        model = torch.nn.DataParallel(model.cuda(), device_ids=gpu)

    # ####################### Loss ####################### #
    if cfg.use_l1:
        criterion_mono = MononizingLoss(cfg)
        criterion_invert = InvertibilityLoss(cfg)
    else:
        criterion_mono = MononizingLossMSE(cfg)
        criterion_invert = InvertibilityLossMSE(cfg)

    # ####################### Optimizer ####################### #
    if cfg.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.base_lr, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr)
    adaptive_lr_sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.factor,
                                                                      patience=cfg.patience, threshold=cfg.threshold,
                                                                      threshold_mode='rel', verbose=True)

    if cfg.weight:
        if os.path.isfile(cfg.weight):
            if main_process(cfg):
                logger.info("=> loading weight '{}'".format(cfg.weight))
            checkpoint = torch.load(cfg.weight, map_location=lambda storage, loc: storage.cuda())
            load_state_dict_roboust(model, checkpoint['state_dict'])
            if main_process(cfg):
                logger.info("=> loaded weight '{}'".format(cfg.weight))
        else:
            if main_process(cfg):
                logger.info("=> no weight found at '{}'".format(cfg.weight))

    if cfg.resume:
        if os.path.isfile(cfg.resume):
            if main_process(cfg):
                logger.info("=> loading checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(cfg.resume, map_location=lambda storage, loc: storage.cuda())
            cfg.start_epoch = checkpoint['epoch']
            load_state_dict_roboust(model, checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_metric = checkpoint['best_metric']
            if main_process(cfg):
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(cfg.resume, checkpoint['epoch']))
        else:
            if main_process(cfg):
                logger.info("=> no checkpoint found at '{}'".format(cfg.resume))

    # ####################### Data Loader ####################### #
    if cfg.data_name == 'Flickr1024':
        from dataset.flickr1024 import Flickr1024
        train_data = Flickr1024(data_root=cfg.data_root, data_list=cfg.train_set,
                                base_resolution=cfg.base_resolution,
                                patch_size=cfg.patch_size, training=True, loop=cfg.loop)
        val_data = Flickr1024(data_root=cfg.data_root, data_list=cfg.val_set,
                              base_resolution=cfg.base_resolution,
                              patch_size=cfg.patch_size_val, training=True, loop=1)
    else:
        raise Exception('Dataset not supported yet'.format(cfg.data_name))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if cfg.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
                                               num_workers=cfg.workers, pin_memory=True, sampler=train_sampler,
                                               worker_init_fn=worker_init_fn)
    if cfg.evaluate:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data) if cfg.distributed else None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.batch_size_val,
                                                 shuffle=False, num_workers=cfg.workers, pin_memory=True,
                                                 drop_last=False,
                                                 worker_init_fn=worker_init_fn, sampler=val_sampler)

    # ####################### Train ####################### #
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
            if cfg.evaluate:
                val_sampler.set_epoch(epoch)

        loss_train, loss_mono, loss_invert = \
            train(train_loader, model, criterion_mono, criterion_invert, optimizer, epoch, cfg)
        epoch_log = epoch + 1
        # Adaptive LR
        if cfg.adaptive_lr:
            adaptive_lr_sheduler.step(loss_train)
        if main_process(cfg):
            logger.info('TRAIN Epoch: {} '
                        'loss_train: {} '
                        'mono_train: {} '
                        'invert_train: {} '
                        .format(epoch_log, loss_train, loss_mono, loss_invert)
                        )
            for m, s in zip([loss_train, loss_mono, loss_invert],
                            ["train/loss", "train/mono", "train/invert"]):
                writer.add_scalar(s, m, epoch_log)

        is_best = False
        if cfg.evaluate and (epoch_log % cfg.eval_freq == 0):
            loss_val, loss_mono, loss_invert, PSNR, SSIM = \
                validate(val_loader, model, criterion_mono, criterion_invert, epoch, cfg)
            if main_process(cfg):
                logger.info('VAL Epoch: {} '
                            'loss_val: {} '
                            'mono_val: {} '
                            'invert_val: {} '
                            'PSNR: {},{},{} '
                            'SSIM: {},{},{} '
                            .format(epoch_log, loss_val, loss_mono, loss_invert, *PSNR, *SSIM)
                            )
                for m, s in zip([loss_val, loss_mono, loss_invert, *PSNR, *SSIM],
                                ["val/loss", "val/mono", "val/invert", "val/PSNR_m", "val/PSNR_l",
                                 "val/PSNR_r", "val/SSIM_m", "val/SSIM_l", "val/SSIM_r"]):
                    writer.add_scalar(s, m, epoch_log)

            # remember best iou and save checkpoint
            is_best = loss_val < best_metric
            best_metric = min(best_metric, loss_val)
        if (epoch_log % cfg.save_freq == 0) and main_process(cfg):
            save_checkpoint_roboust(model,
                                    other_state={
                                        'epoch': epoch_log,
                                        'state_dict': model.state_dict(),
                                        'optimizer': optimizer.state_dict(),
                                        'best_metric': best_metric},
                                    sav_path=os.path.join(cfg.save_path, 'model'),
                                    is_best=is_best
                                    )


def train(train_loader, model, criterion_mono, criterion_invert, optimizer, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    mono_meter, invert_meter = AverageMeter(), AverageMeter()

    model.train()
    end = time.time()
    max_iter = cfg.epochs * len(train_loader)
    for i, (left, right) in enumerate(train_loader):
        current_iter = epoch * len(train_loader) + i + 1
        data_time.update(time.time() - end)
        left = left.cuda(non_blocking=True)
        right = right.cuda(non_blocking=True)

        mononized, restored_l, restored_r = model(left, right)
        mono_loss = criterion_mono(mononized, left)
        invert_loss = criterion_invert(restored_l, restored_r, left, right)
        loss = mono_loss + cfg.lambda_invert * invert_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        for m, x in zip([loss_meter, mono_meter, invert_meter ],
                        [loss, mono_loss, invert_loss
                         ]):
            m.update(x.item(), left.shape[0])
        # Adjust lr
        if cfg.poly_lr:
            current_lr = poly_learning_rate(cfg.base_lr, current_iter, max_iter, power=cfg.power)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % cfg.print_freq == 0 and main_process(cfg):
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        .format(epoch + 1, cfg.epochs, i + 1, len(train_loader),
                                batch_time=batch_time, data_time=data_time,
                                remain_time=remain_time,
                                loss_meter=loss_meter
                                ))
        if main_process(cfg):
            writer.add_scalar('learning_rate', current_lr, current_iter)
            for m, s in zip([loss_meter, mono_meter, invert_meter],
                            ["train_batch/loss", "train_batch/mono", "train_batch/invert"]):
                writer.add_scalar(s, m.val, current_iter)
    return loss_meter.avg, mono_meter.avg, invert_meter.avg


def validate(val_loader, model, criterion_mono, criterion_invert, epoch, cfg):
    loss_meter = AverageMeter()
    mono_meter, invert_meter = AverageMeter(), AverageMeter()
    psnr_meter_mlr, ssim_meter_mlr = [AverageMeter() for _ in range(3)], [AverageMeter() for _ in range(3)]

    psnr_calor = psnr.PSNR(transform=inverse_normalize)
    ssim_calor = ssim.SSIM(transform=inverse_normalize)

    model.eval()
    with torch.no_grad():
        for i, (left, right) in enumerate(val_loader):
            left = left.cuda(non_blocking=True)
            right = right.cuda(non_blocking=True)

            mononized, restored_l, restored_r = model(left, right)
            mono_loss = criterion_mono(mononized, left)
            invert_loss = criterion_invert(restored_l, restored_r, left, right)
            loss = mono_loss + cfg.lambda_invert * invert_loss

            batch_psnr_m, batch_psnr_l, batch_psnr_r = \
                psnr_calor(mononized, left), psnr_calor(restored_l, left), psnr_calor(restored_r, right)
            batch_ssim_m, batch_ssim_l, batch_ssim_r = \
                ssim_calor(mononized, left), ssim_calor(restored_l, left), ssim_calor(restored_r, right)
            if cfg.distributed:
                loss = reduce_tensor(loss, cfg)
                mono_loss = reduce_tensor(mono_loss, cfg)
                invert_loss = reduce_tensor(invert_loss, cfg)
                batch_psnr_m = reduce_tensor(batch_psnr_m, cfg)
                batch_psnr_l = reduce_tensor(batch_psnr_l, cfg)
                batch_psnr_r = reduce_tensor(batch_psnr_r, cfg)
                batch_ssim_m = reduce_tensor(batch_ssim_m, cfg)
                batch_ssim_l = reduce_tensor(batch_ssim_l, cfg)
                batch_ssim_r = reduce_tensor(batch_ssim_r, cfg)

            for m, x in zip([loss_meter, mono_meter, invert_meter, *psnr_meter_mlr, *ssim_meter_mlr],
                            [loss, mono_loss, invert_loss, batch_psnr_m, batch_psnr_l, batch_psnr_r,
                             batch_ssim_m, batch_ssim_l, batch_ssim_r]):
                m.update(x.item(), left.shape[0])

        # Visualize after validation
        if main_process(cfg):
            sampleEncoded = torchvision.utils.make_grid(inverse_normalize(mononized).clamp(0.0, 1.0))
            sampleResL = torchvision.utils.make_grid(inverse_normalize(restored_l).clamp(0.0, 1.0))
            sampleResR = torchvision.utils.make_grid(inverse_normalize(restored_r).clamp(0.0, 1.0))
            sampleRight = torchvision.utils.make_grid(inverse_normalize(right).clamp(0.0, 1.0))
            writer.add_image('sample_results/mononized', sampleEncoded, epoch + 1)
            writer.add_image('sample_results/restored_l', sampleResL, epoch + 1)
            writer.add_image('sample_results/restored_r', sampleResR, epoch + 1)
            writer.add_image('sample_results/right', sampleRight, epoch + 1)

    return loss_meter.avg, mono_meter.avg, invert_meter.avg, \
           [m.avg for m in psnr_meter_mlr], [m.avg for m in ssim_meter_mlr]


if __name__ == '__main__':
    main()
