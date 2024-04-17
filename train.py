import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from model import HandNet
import time
import cv2
import numpy as np
import torch
from torchvision.transforms import functional
from lib.datasets.dexycb import DexYCB
from torch.utils.data.dataloader import DataLoader
from lib.utils.config import CN
from args import get_intr

# tune multi-threading params
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

cv2.setNumThreads(0)

datapath = '/mnt/public/datasets'

# LOAD the data
DEXYCB_3D_CONFIG_TRAIN = dict(
    DATA_MODE="3D",
    DATA_ROOT=datapath,
    DATA_SPLIT="train",
    SETUP="s0",
    USE_LEFT_HAND=False,
    FILTER_INVISIBLE_HAND=True,
    TRANSFORM=dict(
        TYPE="SimpleTransform3DMANO",
        AUG=True,
        SCALE_JIT=0.125,
        COLOR_JIT=0.3,
        ROT_JIT=30,
        ROT_PROB=1.0,
        OCCLUSION=True,
        OCCLUSION_PROB=0.5,
    ),
    DATA_PRESET=dict(
        USE_CACHE=True,
        BBOX_EXPAND_RATIO=1.7,
        IMAGE_SIZE=(256, 256),
        CENTER_IDX=0,
    ),
)

DEXYCB_3D_CONFIG_TEST = dict(
    DATA_MODE="3D",
    DATA_ROOT=datapath,
    DATA_SPLIT="test",
    SETUP="s0",
    USE_LEFT_HAND=False,
    FILTER_INVISIBLE_HAND=True,
    TRANSFORM=dict(
        TYPE="SimpleTransform3DMANO",
        AUG=True,
        SCALE_JIT=0.125,
        COLOR_JIT=0.3,
        ROT_JIT=30,
        ROT_PROB=1.0,
        OCCLUSION=True,
        OCCLUSION_PROB=0.5,
    ),
    DATA_PRESET=dict(
        USE_CACHE=True,
        BBOX_EXPAND_RATIO=1.7,
        IMAGE_SIZE=(256, 256),
        CENTER_IDX=0,
    ),
)

cfg_train = CN(DEXYCB_3D_CONFIG_TRAIN)
cfg_test = CN(DEXYCB_3D_CONFIG_TEST)

train_dataset = DexYCB(cfg_train)
train_data = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataset = DexYCB(cfg_test)
test_data = DataLoader(test_dataset, batch_size=4, shuffle=True)


print("Train Dataset Len: ", len(train_dataset))
print("Test Dataset Len: ", len(test_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HandNet()
model.train()
model = model.cuda()

# hm[-1], heat_map use MSE
#  so3, so3 use MSE
#  beta, beta use MSE
#  joint_root,
#  bone_vis / 2
loss_hm = nn.MSELoss()
loss_so3 = nn.MSELoss()
loss_beta = nn.MSELoss()
loss_joint_root = nn.MSELoss()

l_hm, l_so3, l_beta, l_joint_root = 1, 1, 1, 1

lr = 1e-5
epoches = 1
train_step, test_step = 0, 0
optimizer = torch.optim.Adam(model.parameters(), lr)


def training(train_data, test_data, epoches):
    for i in range(epoches):
        print("--------------第{}次训练--------------".format(i))
        img = []
        for data in train_data:
            img.append(data['image'])
        print(img.shape)
        img = torch.tensor(img).unsqueeze(0).to(device)
        hm, so3, beta, joint_root, bone_vis = model(img, intr)
        print(joint_root.shape)


# training(train_dataset, test_dataset, epoches)
data = next(iter(train_data))

print(data['image'].shape)
imgs = data['image'].to(device)
intr = get_intr(imgs)
hm, so3, beta, joint_root, bone_vis = model(imgs, intr)
print(hm.shape, so3.shape, beta.shape, joint_root.shape)
print(joint_root)
print(data)
# training(epoches)


# dict_keys(['rot_rad', 'rot_mat3d', 'affine', 'image', 'target_bbox_center', 'target_bbox_scale', 'target_joints_2d',
# 'target_joints_vis', 'image_path', 'affine_postrot', 'target_cam_intr', 'target_joints_3d', 'target_verts_3d',
# 'target_mano_pose', 'target_mano_shape', 'idx', 'image_full', 'cam_center', 'bbox_center', 'bbox_scale', 'cam_intr',
# 'joints_2d', 'joints_3d', 'verts_3d', 'joints_vis', 'joints_uvd', 'verts_uvd', 'mano_pose', 'mano_shape', 'raw_size'])