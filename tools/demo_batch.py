#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
# labels_res_path = '/home/zzc/DKZN/MDect/Yolox/YOLOX/YOLOX_outputs/yolox_m/label_res'
labels_res_path = '/media/zzc/训练专用盘/person_incar/testImg/testRes'


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    # 处理数据的模式
    parser.add_argument(
        "demo",
        default="image",
        help="demo type, eg. image, video and webcam"
    )
    #
    parser.add_argument(
        "-expn", "--experiment-name", type=str, default=None
    )
    # 模型名称
    parser.add_argument(
        "-n", "--name", type=str, default=None, help="model name"
    )

    # 处理数据的路径
    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument(
        "--camid",
        type=int,
        default=0,
        help="webcam demo camera id"
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        default=None,
        type=str,
        help="ckpt for eval"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument(
        "--conf",
        default=None,
        type=float,
        help="test conf"
    )
    parser.add_argument(
        "--nms",
        default=None,
        type=float,
        help="test nms threshold"
    )
    parser.add_argument(
        "--tsize",
        default=None,
        type=int,
        help="test img size"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            trt_file=None,
            decoder=None,
            device="cpu",
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, txt_save_path, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, txt_save_path, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        curLine = image_name.strip().split('/')
        print(curLine)
        txtName = "".join(list(curLine[-1])[0:-4]) + '.txt'
        outputs, img_info = predictor.inference(image_name)
        # ---- txt save path ----
        txt_save_path = os.path.join(labels_res_path, os.path.basename(txtName))
        print("txt_save_path:", txt_save_path)
        # ---- result image ----
        result_image = predictor.visual(outputs[0], img_info, txt_save_path, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    # 检测器
    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


def gain_test_img(testTxtPath, testImgPath, saveImgPath):
    # 对测试集进行可是化操作
    with open(testTxtPath, 'r') as f:
        testTxtLines = f.readlines()
        for i in testTxtLines:
            curLine = i.strip().split(" ")
            testImgName = testImgPath + "/" + curLine[0] + ".png"
            testImg = cv2.imread(testImgName)
            saveImgName = saveImgPath + "/" + curLine[0] + ".png"
            print(saveImgName)
            cv2.imwrite(saveImgName, testImg)


if __name__ == "__main__":
    # 处理单个数据来源
    # args = make_parser().parse_args()
    # args.path = '/home/zzc/DKZN/datasets/UA-DETRAC/DETRAC-train-val-data/val/MVI_20052/img1/img00024.jpg'
    # args.device = 'gpu'
    # args.conf = 0.3
    # args.name = 'yolox-s'
    # args.ckpt = '/home/zzc/DKZN/MDect/Yolox/YOLOX/YOLOX_outputs/yolox_voc_s_UA-DETRAC_yolox_s/latest_ckpt.pth'
    # args.nms = 0.5
    # args.tsize = 640
    # args.save_result = True
    # print(args)
    #
    # exp = get_exp(args.exp_file, args.name)
    # main(exp, args)

    # 清空文件夹下所有的标签 txt 文件
    for i in os.listdir(labels_res_path):
        os.remove(labels_res_path + "/" + i)

    # # 清空检测结果图片文件下 images 文件
    # for i in os.listdir("../YOLOX_outputs/")

    testTxtPath = '/media/zzc/训练专用盘/person_incar/ImageSets/Main/test.txt'
    testImgPath = '/media/zzc/训练专用盘/person_incar/JPEGImages'
    saveImgPath = '/media/zzc/训练专用盘/person_incar/testImg/testOra'
    # gain_test_img(testTxtPath, testImgPath, saveImgPath)

    # 处理多个数据来源
    args = make_parser().parse_args()
    # LG
    # args.path = '/media/zzc/Backup Plus/数据集/车辆目标检测/临港相关数据集/0907/云端路站上行-22号机/images-ydlsx-0907/'
    # defect

    args.path = saveImgPath
    args.device = 'gpu'
    args.conf = 0.30
    args.name = 'yolox-m'
    args.devices = 0
    # LG
    # args.ckpt = '/home/zzc/DKZN/MDect/Yolox/YOLOX/YOLOX_outputs/yolox_voc_m_LG/latest_ckpt.pth'
    # defect
    args.ckpt = '/home/zzc/DKZN/MDect/Yolox/YOLOX/YOLOX_outputs/yolox_m_incar_1000_map50_0.810_map95_0.461/best_ckpt.pth'
    args.nms = 0.50
    args.tsize = 1280
    args.save_result = True
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)

    ##### 使用图片进行测试 #####
    # python3 tools/demo.py image -n yolox-s -c /home/zzc/DKZN/MDect/Yolox/Yolox相关内容/7个模型权重/pt文件/yolox_s.pth --path assets/dog.jpg --conf 0.3 --nms 0.5 --tsize 640 --save_result --device [gpu]
    # python3 tools/demo.py image -n yolox-s -c /home/zzc/DKZN/MDect/Yolox/Yolox相关内容/7个模型权重/pt文件/yolox_s.pth --path /home/zzc/DKZN/test_videos/1.png --conf 0.3 --nms 0.5 --tsize 640 --save_result --device [gpu]
    # python3 tools/demo_batch.py image -n yolox-s -c /home/zzc/DKZN/MDect/Yolox/YOLOX/YOLOX_outputs/yolox_voc_s_UA-DETRAC_yolox_s/latest_ckpt.pth --path /home/zzc/DKZN/test_videos/Picture/1.bmp --conf 0.3 --nms 0.5 --tsize 640 --save_result --device [gpu]
    # python3 tools/demo_batch.py image
    ##### 使用视频进行测试 #####
    # python3 tools/demo.py video -n yolox-s -c /home/zzc/DKZN/MDect/Yolox/YOLOX/YOLOX_outputs/yolox_voc_s_UA-DETRAC_yolox_s/latest_ckpt.pth --path /home/zzc/DKZN/test_videos/20210902-180000-20210902-190000.avi --conf 0.3 --nms 0.5 --tsize 640 --save_result --device [gpu]
