#!/usr/bin/env python
# coding: utf-8

# ---- Library import ----
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA  # SWA(随机权重平均)——一种全新的模型优化方法

# ---- My utils ----
from models import *
from utils.arguments import *
from utils.data_augmentation import data_augmentation_selector
from utils.dataload import dataset_selector
from utils.training import *

np.set_printoptions(precision=4)  # 设置浮点数的精度
train_aug, train_aug_img, val_aug = data_augmentation_selector(args.data_augmentation, args.img_size, args.crop_size)

train_dataset, val_dataset = dataset_selector(train_aug, train_aug_img, val_aug, args)  # 此处的训练集和验证集是经过常规的图像增强操作的；
# 并且此处的的验证集的数量是训练集的15%
if args.dataset == "mnms_and_entropy" or args.dataset == "mnms_and_entropy_and_weakly":
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, collate_fn=train_dataset.masks_collate
    )  # 此处是训练集的准备
else:
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

for i in val_loader:
    print(len(i))


# for i in val_loader:
#     print(len(i))
#
#     print(i[0])