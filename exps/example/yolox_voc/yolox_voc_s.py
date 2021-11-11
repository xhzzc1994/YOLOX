# encoding: utf-8
import os

import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 2  # 修改类别
        self.depth = 0.67  # yolox-s:0.33 # yolox-x:1.33 # yolox-m:0.67
        self.width = 0.75  # yolox-s:0.50 # yolox-x:1.25 # yolox-m:0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            VOCDetection,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = VOCDetection(
            # data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            # image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
            # 人头数据集
            # data_dir="/home/zzc/DKZN/MDect/Yolox/YoloxRela/head",
            # UA-DETRAC数据集
            # data_dir="/media/zzc/Backup Plus/数据集/车辆目标检测/UA-DETRAC",  # 修改训练集信息
            # LG数据集
            # data_dir="/media/zzc/Backup Plus/数据集/车辆目标检测/LG",  # 修改训练集信息
            # 缺陷检测数据集
            # data_dir="/media/zzc/Backup Plus/比赛相关/工业缺陷检测/defect-detecting",  # 修改训练集信息
            # 数据增强后的缺陷检测数据集
            # data_dir="/media/zzc/Backup Plus/比赛相关/工业缺陷检测/defect-imgaug",
            # LG数据集3种类别
            # data_dir="/media/zzc/Backup Plus/数据集/车辆目标检测/临港相关数据集/LG",
            # 车内客流检测
            data_dir="/media/zzc/训练专用盘/person_incar",
            image_sets=[('train')],  # 修改训练集信息
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=100,  # 表示每张图片的最多目标数量
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=100,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import VOCDetection, ValTransform

        valdataset = VOCDetection(
            # data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            # image_sets=[('2007', 'test')],
            # 人头数据集
            # data_dir="/home/zzc/DKZN/MDect/Yolox/YoloxRela/head",
            # UA-DETRAC数据集
            # data_dir="/media/zzc/Backup Plus/数据集/车辆目标检测/UA-DETRAC",  # 修改验证集信息
            # LG车辆检测数据集
            # data_dir="/media/zzc/Backup Plus/数据集/车辆目标检测/LG",
            # 缺陷检测数据集
            # data_dir="/media/zzc/Backup Plus/比赛相关/工业缺陷检测/defect-detecting",  # 修改验证集信息
            # 数据增强后的缺陷检测数据集
            # data_dir="/media/zzc/Backup Plus/比赛相关/工业缺陷检测/defect-imgaug",
            # LG数据集3种类别
            # data_dir="/media/zzc/Backup Plus/数据集/车辆目标检测/临港相关数据集/LG",
            # 车内客流检测
            data_dir="/media/zzc/训练专用盘/person_incar",
            image_sets=[('val')],  # 修改验证集信息
            img_size=self.test_size,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import VOCEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = VOCEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator
