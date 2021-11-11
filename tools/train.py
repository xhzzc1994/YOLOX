#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import Trainer, launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, configure_omp


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument(
        "-expn",
        "--experiment-name",
        type=str,
        default=None
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="model name"
    )

    # distributed
    parser.add_argument(
        "--dist-backend",
        default="nccl",
        type=str,
        help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=8,
        help="batch size"
    )
    parser.add_argument(
        "-d",
        "--devices",
        default=0,
        type=int,
        help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default="/home/zzc/DKZN/MDect/Yolox/YOLOX/exps/example/yolox_voc/yolox_voc_s.py",
        type=str,
        help="plz input your expriment description file",
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="resume training"
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        default=None,
        type=str,
        help="checkpoint file"
    )
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines",
        default=1,
        type=int,
        help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank",
        default=0,
        type=int,
        help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True
    print("Start training!")
    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    # python3 tools/train.py
    # -f exps/example/yolox_voc/yolox_voc_s.py
    # -d 0
    # -b 2
    # -c /home/zzc/DKZN/MDect/Yolox/YoloxRela/7-pt/pt/yolox_m.pth
    args.exp_file = "exps/example/yolox_voc/yolox_voc_s.py"
    args.devices = 0
    args.batch_size = 2
    # args.ckpt = "/home/zzc/DKZN/MDect/Yolox/YoloxRela/7-pt/pt/yolox_m.pth"
    args.ckpt = "/home/zzc/DKZN/MDect/Yolox/YOLOX/YOLOX_outputs/yolox_m_incar_300_map50_0.803_map95_0.428/last_epoch_ckpt.pth"
    args.experiment_name = "yolox_m_incar_1000"
    args.resume = True
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )

    # 人头数据集训练
    # python3 tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 0 -b 16 -c /home/zzc/DKZN/MDect/Yolox/YoloxRela/7-pt/pt/yolox_s.pth
    # 车辆检测数据集训练
    # python3 tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 0 -b 8 -c /home/zzc/DKZN/MDect/Yolox/YoloxRela/7-pt/pt/yolox_s.pth
    # LG车辆数据集训练
    # python3 tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 0 -b 8 -c /home/zzc/DKZN/MDect/Yolox/YoloxRela/7-pt/pt/yolox_x.pth

    # python3 tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 0 -b 2 -c /home/zzc/DKZN/MDect/Yolox/YoloxRela/7-pt/pt/yolox_m.pth
    # --------------------------------------------------------------
    # model: trainModel_2
    # input_size: (1280, 1280)
    # max_epoch: 30
    # no_aug_epochs: 15
    # map_5095: 0.47405431287503763
    # map_50: 0.809406640486575
    # tsize: (1280, 1280)
    # test_conf = 0.01
    # nmsthre = 0.65
    # --------------------------------------------------------------
    # 2021-09-12 02:05:22 | INFO     | yolox.core.trainer:315 -
    # Average forward time: 52.49 ms, Average NMS time: 0.95 ms, Average inference time: 53.45 ms

    # python3 tools/train.py -f exps/example/yolox_voc/yolox_voc_m_LG.py -d 0 -b 2 -c /home/zzc/DKZN/MDect/Yolox/YoloxRela/7-pt/pt/yolox_m.pth
    # --------------------------------------------------------------
    # model: trainModel_3
    # input_szie: (1280, 1280)
    # max_epoch: 300
    # no_aug_epochs: 50
    # map_5095: 0.6649360416489523
    # map_50: 0.902390184852291
    # tsize: (1280, 1280)
    # test_conf = 0.01
    # nmsthre = 0.65
    # --------------------------------------------------------------
    # 2021-09-13 04:48:27 | INFO     | yolox.core.trainer:315 -
    # Average forward time: 52.59 ms, Average NMS time: 0.93 ms, Average inference time: 53.52 ms

    # python3 tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 0 -b 2 -c /home/zzc/DKZN/MDect/Yolox/YoloxRela/7-pt/pt/yolox_m.pth

    # FileNotFoundError: [Errno 2] No such file or directory: '/media/zzc/Backup Plus/数据集/车辆目标检测/临港相关数据集/Annotations/杞青路站下行-8号机-20210907-160000-20210907-170000_00
