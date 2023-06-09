# -*- coding: utf-8 -*-
# @Time : 2023/6/9 14:52
# @Author : zihua.zeng
# @File : inference_edz.py

import os
import sys
import torch
import random
import numpy as np
import shutil
from glob import glob

from config import Config
from models.maniqa import MANIQA
from tqdm import tqdm
from torchvision import transforms
from validate_pr import setup_seed, Image
from utils.inference_process import ToTensor, Normalize

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def inference(net, image_path, output_path=None, output_split=None):
    if os.path.isfile(image_path):

        Img = Image(image_path=image_path,
                    transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
                    img_size=config.img_size,
                    num_crops=config.num_crops)

        avg_score = 0
        for i in tqdm(range(config.num_crops)):
            with torch.no_grad():
                patch_sample = Img.get_patch(i)
                patch = patch_sample['d_img_org'].cuda()
                patch = patch.unsqueeze(0)
                score = net(patch)
                avg_score += score
        cur_score = avg_score / config.num_crops
        print(cur_score)
        return

    # 文件夹
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    thres = None
    if output_split is not None:
        vs = output_split.split(",")
        vs = [float(v) for v in vs]
        thres = {}
        for v in vs:
            sub_folder = "tres_" + str(v)
            thres[v] = sub_folder
            os.makedirs(os.path.join(output_path, sub_folder), exist_ok=True)

    image_paths = glob(os.path.join(image_path, "*"))
    for p in image_paths:
        suffix = os.path.splitext(os.path.basename(p))[-1]
        if suffix.lower() not in [".jpg", ".png"]:
            continue

        try:
            Img = Image(image_path=p,
                        transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
                        img_size=config.img_size,
                        num_crops=config.num_crops)
        except:
            continue

        avg_score = 0
        for i in tqdm(range(config.num_crops)):
            with torch.no_grad():
                patch_sample = Img.get_patch(i)
                patch = patch_sample['d_img_org'].cuda()
                patch = patch.unsqueeze(0)
                score = net(patch)
                avg_score += score
        cur_score = avg_score / config.num_crops

        print(os.path.basename(p), cur_score)

        if thres is not None:
            intervals = [0.0] + list(thres.keys())

            sf = None
            for i in range(len(intervals) - 1):
                if intervals[i] <= cur_score < intervals[i + 1]:
                    sf = thres[intervals[i + 1]]

            if sf is None:
                continue
            shutil.copy(p, os.path.join(output_path, sf, str(cur_score) + "_" + os.path.basename(p)))


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    config = Config({

        # valid times
        "num_crops": 8,

        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.8,

        # checkpoint path
        "ckpt_path": "weights/ckpt_koniq10k.pt",
    })

    # model defination
    net = MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
                 patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
                 depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)

    net.load_state_dict(torch.load(config.ckpt_path), strict=False)
    net = net.cuda()
    net.eval()

    image_path = sys.argv[1]
    output_path = None
    output_split = None
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]

    if len(sys.argv) == 4:
        output_split = sys.argv[3]

    inference(net, image_path, output_path, output_split)
