# -*- coding: utf-8 -*-
# @Time : 2023/6/6 14:32
# @Author : zihua.zeng
# @File : test.py

import os
import shutil
import random
from glob import glob

import torch
from tqdm import tqdm


def format_lq_ds():
    input_path = "/home/zihua/data_warehouse/low_quality/al_sampler_0531_edz"
    out_path = "/home/zihua/data_warehouse/low_quality/iqa_0606/"

    normal_image = glob(input_path + "/training/Normal/*.jpg") + \
                   glob(input_path + "/validation/Normal/*.jpg")

    ablight_image = glob(input_path + "/training/Uneven/*.jpg") + \
                    glob(input_path + "/validation/Uneven/*.jpg")

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, "image"), exist_ok=True)

    label_info = [(os.path.basename(x), 0.0) for x in normal_image] + \
                 [(os.path.basename(x), 1.0) for x in ablight_image]

    random.shuffle(label_info)

    for item in tqdm(normal_image + ablight_image):
        shutil.copy(item, os.path.join(out_path, "image", os.path.basename(item)))

    fp = open(os.path.join(out_path, "label.txt"), "w")
    for item in label_info:
        fp.write("%s, %.2f\n" % (item[0], item[1]))
    fp.close()


def test_datapipe():
    from torchvision import transforms
    from data.koniq10k.koniq10k import Koniq10k
    from utils.process import split_dataset_koniq10k
    from utils.process import RandRotation, RandHorizontalFlip
    from utils.process import RandCrop, ToTensor, Normalize

    train_name, val_name = split_dataset_koniq10k(
        txt_file_name="data/koniq10k/koniq10k_label.txt",
        split_seed=1
    )
    dis_train_path = "data/koniq10k/image"
    dis_val_path = "data/koniq10k/image"
    label_train_path = "data/koniq10k/koniq10k_label.txt"
    label_val_path = "data/koniq10k/koniq10k_label.txt"

    aug_type = "default"

    if aug_type == "default":
        tfs = transforms.Compose([RandCrop(patch_size=320),
                                  Normalize(0.5, 0.5),
                                  RandHorizontalFlip(prob_aug=0.7),
                                  ToTensor()])
    elif aug_type == "custom":
        tfs = transforms.Compose([
            transforms.RandomResizedCrop(320),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    # data load
    train_dataset = Koniq10k(
        dis_path=dis_train_path,
        txt_file_name=label_train_path,
        list_name=train_name,
        transform=tfs,
        keep_ratio=1.0
    )

    item = next(iter(train_dataset))


def test_model():
    from models.maniqa import MANIQA
    net = MANIQA(embed_dim=768,
                 num_outputs=1, dim_mlp=768,
                 patch_size=16, img_size=384, window_size=4,
                 depths=[2, 2],
                 num_heads=[4, 4],
                 num_tab=2,
                 scale=0.8)
    dummy_inp = torch.randn(1, 3, 384, 384)
    out = net(dummy_inp)


if __name__ == '__main__':
    # format_lq_ds()
    # test_datapipe()
    test_model()
    pass
