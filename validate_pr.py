# -*- coding: utf-8 -*-
# @Time : 2023/6/6 18:32
# @Author : zihua.zeng
# @File : validate_ablight.py
#
# 验证模型在二分类任务上的 Precision 和 Recall
#


import os
import torch
import numpy as np
import random
import cv2

from torchvision import transforms
from models.maniqa import MANIQA
from torch.utils.data import DataLoader
from config import Config
from utils.inference_process import ToTensor, Normalize
from tqdm import tqdm
from glob import glob
from torchmetrics.classification import MulticlassRecallAtFixedPrecision, MulticlassRecall, MulticlassPrecision, \
    MulticlassConfusionMatrix

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Image(torch.utils.data.Dataset):
    def __init__(self, image_path, transform, img_size, num_crops=20):
        super(Image, self).__init__()
        self.img_name = image_path.split('/')[-1]
        self.img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = np.array(self.img).astype('float32') / 255

        h, w, c = self.img.shape
        if h <= img_size:
            self.img = cv2.resize(self.img, (img_size + 32, img_size + 32))
            h, w, c = self.img.shape

        self.img = np.transpose(self.img, (2, 0, 1))
        self.transform = transform

        print(self.img.shape)
        new_h = img_size
        new_w = img_size

        try:
            self.img_patches = []
            for i in range(num_crops):
                top = np.random.randint(0, h - new_h)
                left = np.random.randint(0, w - new_w)
                patch = self.img[:, top: top + new_h, left: left + new_w]
                self.img_patches.append(patch)
        except:
            from IPython import embed
            embed()

        self.img_patches = np.array(self.img_patches)

    def get_patch(self, idx):
        patch = self.img_patches[idx]
        sample = {'d_img_org': patch, 'score': 0, 'd_name': self.img_name}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    folder_normal_path = "/home/zihua/data_warehouse/low_quality/al_sampler_0531_edz/validation/Normal"
    fns = glob("%s/*.jpg" % (folder_normal_path))

    foler_ablight_path = "/home/zihua/data_warehouse/low_quality/al_sampler_0531_edz/validation/Uneven"
    fns_ablight = glob("%s/*.jpg" % (foler_ablight_path))

    # config file
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
        "ckpt_path": "output/models/Koniq10k/koniq10k-base_s20/ckpt_product_e227.pt",
    })

    # model defination
    net = MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
                 patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
                 depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)

    net.load_state_dict(torch.load(config.ckpt_path), strict=False)
    net = net.cuda()
    net.eval()

    output_all = []
    target_all = []
    count = 0
    for label, cur_fns in [(0, fns), (1, fns_ablight)]:
        for f in cur_fns:

            # data load
            Img = Image(image_path=f,
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
            count += 1
            print("Image {} score: {}, Progress {} / {}".format(Img.img_name, cur_score, count,
                                                                len(fns) + len(fns_ablight)))

            cur_score = cur_score.detach().cpu().numpy()[0]
            output_all.append([1 - cur_score, cur_score])
            target_all.append(label)

    num_classes = 2
    output_all = torch.from_numpy(np.array(output_all))
    target_all = torch.from_numpy(np.array(target_all))

    precssss = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    prec_arr = []
    recall_arr = []
    threshold_arr = []
    for prec in precssss:
        mcrafp = MulticlassRecallAtFixedPrecision(num_classes=num_classes, min_precision=prec, thresholds=None)

        recalls, thresholds = mcrafp(output_all, target_all)
        # print("mcr recall is:", recalls[-1])
        # print("mcr threshold is:", thresholds[-1])

        pred_all = (output_all[:, -1] >= thresholds[-1]).int()
        ##print("pred_all shape:", pred_all.shape)

        # mcf1s = MulticlassF1Score(num_classes=args.num_classes, average=None).to(device)
        confmat = MulticlassConfusionMatrix(num_classes=num_classes)
        mcr = MulticlassRecall(num_classes=num_classes, average=None)
        mcp = MulticlassPrecision(num_classes=num_classes, average=None)

        precision = mcp(pred_all, target_all)[-1]
        recall = mcr(pred_all, target_all)[-1]

        prec_arr.append(precision)
        recall_arr.append(recall)
        threshold_arr.append(thresholds[-1])

        # print("precision is {}, recall is {}, threshold is {}".format(precision, recall, thresholds[-1]))

        # f1 = mcf1s(output_all, target_all)[-1]
        cmetrics = confmat(pred_all, target_all)
        print("conf metrix: %.2f" % prec, cmetrics)

    print("Fixed Precision, Precision, Recall, Threshold")
    for i, j, k, l in zip(precssss, prec_arr, recall_arr, threshold_arr):
        print("%.4f, %.4f, %.4f, %.6f" % (i, j, k, l))

    # print("Image {} score: {}".format(Img.img_name, avg_score / config.num_crops))
