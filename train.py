import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data.dataloader import DataLoader

from torchvision.transforms import functional
from torchvision import transforms

from lib.datasets.dexycb import DexYCB
from lib.utils.config import CN
from model import HandNet

from torch.utils.tensorboard import SummaryWriter


# tune multi-threading params
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

cv2.setNumThreads(0)

datapath = '/mnt/public/datasets'
img_w = 256
img_h = 256

writer = SummaryWriter('runs/HandNet')

# LOAD the data
# dict_keys(['rot_rad', 'rot_mat3d', 'affine', 'image', 'target_bbox_center', 'target_bbox_scale', 'target_joints_2d',
    # 'target_joints_vis', 'image_path', 'affine_postrot', 'target_cam_intr', 'target_joints_3d', 'target_verts_3d',
    # 'target_mano_pose', 'target_mano_shape', 'idx', 'image_full', 'cam_center', 'bbox_center', 'bbox_scale', 'cam_intr',
    # 'joints_2d', 'joints_3d', 'verts_3d', 'joints_vis', 'joints_uvd', 'verts_uvd', 'mano_pose', 'mano_shape', 'raw_size'])
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
        IMAGE_SIZE=(img_w, img_h),
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
        IMAGE_SIZE=(img_w, img_h),
        CENTER_IDX=0,
    ),
)

def hm_graph(batch_size, joints_2d):
    #define the resize shape
    hm_target_batch = list()
    w = 64
    h = 64
    for i in range(batch_size):
        hm_target = list()
        for j in range(joints_2d.shape[1]):
            
            img = np.zeros((256, 256), dtype=np.uint8)  #(H, W)
            center = tuple(joints_2d[i, j, :].numpy().astype(np.uint8))
            cv2.circle(img, center, 4, 255, -1)   # add joints
            heat_map = torch.tensor(img).unsqueeze(0)
            #resize, the interpolation mode is NEAREST, may be others
            heat_map = functional.resize(heat_map, (h, w), 
                                         interpolation=transforms.InterpolationMode.NEAREST) 
            
            # to visualize , can be deleted
            # heat_map = heat_map.permute(1,2,0)
            # graph = heat_map.numpy().squeeze(2).copy()
            # print(graph.shape)
            # cv2.imwrite('./joint_image/joint_image{}.jpg'.format(j), graph)
            
            hm_target.append(heat_map)
        #heat_map = heat_map.permute(1,2,0) #(H, W, C), to visualize , can be deleted
        
        hm_stack = torch.stack(hm_target)
        hm_target_batch.append(hm_stack)
        
        # used to visualize , can be deleted
        # img = heat_map.numpy().copy()  
        # print(heat_map.shape)
        # cv2.imwrite('./joint_image/joint_image{}.jpg'.format(i), img)
    hm_target_batch = torch.stack(hm_target_batch).squeeze(2).squeeze(3).type(torch.float32)
    return hm_target_batch
            
            
def collation_fn_for_dict(batch):

    if not isinstance(batch, list):
        # Use list to align with the following format
        # This is the case where only 1 sample is provided.
        batch = [batch]

    batch_concat = dict()

    keys_to_collate = [
        'image',
        'target_joints_2d',
        'target_joints_3d',
        'target_mano_pose',
        'target_cam_intr',
    ]

    # for each in batch[0]:  # collate all keys
    for each in keys_to_collate:
        if isinstance(batch[0][each], np.ndarray) and not isinstance(batch[0][each][0], str):
            batch_concat[each] = np.concatenate([batch[i][each] for i in range(len(batch))], axis=0)
            batch_concat[each] = torch.Tensor(batch_concat[each])
        else:
            batch_concat[each] = [batch[i][each] for i in range(len(batch))]

    return batch_concat


def training(train_data, test_data, epoches):
    
    for i in range(epoches):
        print("--------------第{}次训练--------------".format(i))
        model.train()
        train_loss = 0
        test_loss = 0
        for idx, data in enumerate(train_data):
            imgs = data['image'].to(device)
            joints_2d = data['joints_2d']
            hm_target = hm_graph(batch_size, joints_2d).to(device)
            so3_target = torch.tensor(data['target_mano_pose'], device=device)
            joint_root_target = torch.tensor(data['target_joints_3d'][:,0:1,:], device=device)
            intr = torch.tensor(data['target_cam_intr'], device=device)

            optimizer.zero_grad()
            
            hm, so3, beta, joint_root, bone_vis = model(imgs, intr)
            
            so3 = torch.split(so3, 16, dim=1)
            so3 = torch.stack(so3, dim=2)
            
            print("target_joints_2d shape: {}".format(hm_target.shape))
            print("target_so3 shape: {}".format(so3_target.shape))
            print("target_joint_root shape: {}".format(joint_root_target.shape))
            print("joints_2d shape: {}".format(hm.shape))
            print("so3 shape: {}".format(so3.shape))   
            print("joint_root shape: {}".format(joint_root.shape))
            
            loss = l_hm * loss_hm(hm_target, hm) + l_so3 * loss_so3(so3_target, so3) + \
                l_joint_root * loss_hm(joint_root_target, joint_root)
                
            train_loss = loss.item()
            
            loss.backward()
            optimizer.step()
            print("Train loss: {}".format(train_loss))

        
        # test
        with torch.no_grad():
            model.eval()
            for data in test_data:
                imgs = data['image'].to(device)

                hm_target = data['target_joints_2d'].to(device)
                so3_target = data['target_joints_3d'].to(device)
                joint_root_target = data['target_mano_pose'].to(device)
                intr = data['target_cam_intr'].to(device)
                hm, so3, beta, joint_root, bone_vis = model(imgs, intr)

                loss = l_hm * loss_hm(hm_target, hm) + l_so3 * loss_so3(so3_target, so3) + \
                    l_joint_root * loss_hm(joint_root_target, joint_root)

                test_loss += loss
        print("Test loss: {}".format(test_total_loss))
        writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : train_loss, 'Validation' : test_loss },
                            epoches)



if __name__ == '__main__':
    cfg_train = CN(DEXYCB_3D_CONFIG_TRAIN)
    cfg_test = CN(DEXYCB_3D_CONFIG_TEST)
    batch_size = 64
    train_dataset = DexYCB(cfg_train)
    train_data = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataset = DexYCB(cfg_test)
    test_data = DataLoader(test_dataset, batch_size, shuffle=True)


    print("Train Dataset Len: ", len(train_dataset))
    print("Test Dataset Len: ", len(test_dataset))
    print("Train Dataloader Batch: ", len(train_data))
    print("Test Dataloader Batch: ", len(test_data))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HandNet()
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

    lr = 0.02
    epoches = 20
   
    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    data = next(iter(train_data))
    
    training(train_data, test_data, epoches)

    writer.flush()

# pointing the joints of hand #

# def hm_graph_with_picture(batch_size, imgs, joints_2d):
#     w = 64
#     h = 64
#     for i in range(batch_size):
#         img = imgs[i,:,:,:]
#         img = img.permute((1,2,0)).detach().cpu().numpy().copy()
#         joints_2d = torch.round(joints_2d)

#         for j in range(joints_2d.shape[1]):
#             center = tuple(joints_2d[i, j, :].numpy().astype(int))
#             cv2.circle(img, center, 4, (255,255,255), -1)
#         heat_map = torch.tensor(img)
#         heat_map = heat_map.permute(2,0,1)
#         heat_map = functional.resize(heat_map, (h, w))
#         heat_map = heat_map.permute(1,2,0)
#         img = heat_map.numpy().copy()
#         print(heat_map.shape)
#         cv2.imwrite('./hand_image/hand_image{}.jpg'.format(i), img)
