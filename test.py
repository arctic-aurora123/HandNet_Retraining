import os
import time

# tune multi-threading params
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(0)

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
datapath = '/mnt/public/datasets'
img_w = 256
img_h = 256

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

def hm_graph_target(batch_size, joints_2d, hm):
    #define the resize shape
    # 创建一个figure对象，指定图形的大小
    fig = plt.figure(figsize=(10, 5))

    # 在figure对象中创建两个子图，指定projection为'3d'
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    hm_target_batch = list()
    w = 64
    h = 64
    for i in range(batch_size):
        hm_target = list()
        joints_2d /= 4
        for j in range(joints_2d.shape[1]):
            
            img = np.zeros((w, h), dtype=np.int32)  #(H, W)
            center = joints_2d[i, j, :].numpy().astype(np.int32).tolist()
            img[center[0]][center[1]] = 1
            heat_map = torch.tensor(img).unsqueeze(0)
            #resize, the interpolation mode is NEAREST, may be others
            
            # to visualize , can be deleted
            heat_map = heat_map.permute(1,2,0)
            graph = heat_map.numpy().squeeze(2).copy()
            hm_target.append(heat_map)
        heat_map = heat_map.permute(1,2,0) #(H, W, C), to visualize , can be deleted
        
        hm_stack = torch.stack(hm_target)
        hm_target_batch.append(hm_stack)
        
        #used to visualize
        img_target = heat_map.numpy().copy()  
        ax1.imshow(img_target)
        # cv2.imwrite('./joint_image_target/joint_image_target{}.jpg'.format(i), img)
    hm_target_batch = torch.stack(hm_target_batch).squeeze(2).squeeze(3).type(torch.float32)
    return hm_target_batch


def get_hm_taget_new(pred_hm, joints_2d):
    # by TTT at 2024.4.22
    # TODO: We could remove the for-loop by using advanced indexing techniques,
    #       but we should do it later.
    b, c, h, w = pred_hm.shape
    target_hm = torch.zeros_like(pred_hm)
    # print(target_hm.shape)  # B, 21, 64, 64
    # print(joints_2d.shape)  # B, 21, 2
    joints_2d = torch.tensor(joints_2d / 4, dtype=torch.long)
    for batch_idx in range(b):
        for channel_idx in range(c):
            x, y = joints_2d[batch_idx][channel_idx]
            if 0<= x < w and 0 <= y < h:
                target_hm[batch_idx][channel_idx][y][x] = 1
    # mat = target_hm[batch_idx][channel_idx].clone().detach().cpu().numpy()
    # plt.imshow(mat, cmap ='gray')    
    # plt.show()
        # else: pass - do not draw on the image
    return target_hm
    
def so3_vis(so3, so3_target, img):
    img = img.permute(1,2,0)
    img *= -1
    so3 = so3.clone().detach().cpu().numpy()
    so3_target = so3_target.clone().detach().cpu().numpy()
    img = img.clone().detach().cpu().numpy()

    # 创建一个figure对象，指定图形的大小
    fig = plt.figure(figsize=(10, 5))

    # 在figure对象中创建两个子图，指定projection为'3d'
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133)
    # 在第一个子图中绘制三维曲线
    # 生成曲线的数据
    x1 = so3[:, 0]
    y1 = so3[:, 1]
    z1 = so3[:, 2]

    # 使用plot方法绘制曲线，设置颜色为红色
    ax1.scatter(x1, y1, z1, color='r')

    # 设置x、y和z轴的标签
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # 在第二个子图中绘制三维散点图
    # 生成散点的数据
    x2 = so3_target[:, 0]
    y2 = so3_target[:, 1]
    z2 = so3_target[:, 2]

    # 使用scatter方法绘制散点，设置颜色为蓝色
    ax2.scatter(x2, y2, z2, color='b')

    # 设置x、y和z轴的标签
    ax2.set_xlabel('X')
    
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    ax3.imshow(img)
    # 显示图形
    plt.show()

def hm_vis(hm, hm_target, batch_size):
    hm = hm.clone().detach().cpu().numpy()
    hm_target = hm_target.clone().detach().cpu().numpy()
    
    fig = plt.figure(figsize=(10, 5))
    
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for idx in range(batch_size):
        for pic in range(21):
            ax1.imshow(hm[idx][pic])
            ax2.imshow(hm_target[idx][pic])
            plt.show()
    # mat1 = np.zeros((64,64))
    # mat2 = np.zeros((64,64))
    # for idx in range(batch_size):
    #     for pic in range(21):
    #         mat1 += hm[idx][pic]
    #         ax1.imshow(mat1)
    #         mat2 += hm_target[idx][pic]
    #         ax2.imshow(mat2)
    #     plt.show()    
    
def testing(test_data, epoches):
    
    n_iter = 0
    model.eval()
    for n_epochs in range(epoches):
        # test
        with torch.no_grad():
            
            for data in test_data:
                imgs = data['image'].to(device)
                so3_target = torch.tensor(data['target_mano_pose'], device=device)
                joint_root_target = torch.tensor(data['target_joints_3d'][:,0:1,:], device=device)
                intr = torch.tensor(data['target_cam_intr'], device=device)
                hm, so3, beta, joint_root, bone_vis = model(imgs, intr)
                
                hm_target = get_hm_taget_new(hm, data['target_joints_2d'])

                so3 = torch.split(so3, batch_size, dim=1)
                so3 = torch.stack(so3, dim=2)
                #so3_target = so3_target.reshape(batch_size, 48)
                # for idx in range(batch_size):
                #     print(so3[idx])
                #     print(so3_target[idx])
                #     print("\r\n")
                #     # print(joint_root[idx])
                #     # print(joint_root_target[idx])
                #     # print("\r\n")
                    
                #     so3_vis(so3[idx], so3_target[idx],imgs[idx])
                print(hm.shape)
                print(hm_target.shape)
                hm_vis(hm, hm_target, batch_size)
                # loss_hm_value = loss_hm(hm_target, hm)
                # loss_so3_value = loss_so3(so3_target, so3)
                # loss_joint_root_value = loss_hm(joint_root_target, joint_root)
                # loss = l_hm * loss_hm_value + l_so3 * loss_so3_value + l_joint_root * loss_joint_root_value
        

if __name__ == '__main__':
    cfg_train = CN(DEXYCB_3D_CONFIG_TRAIN)
    cfg_test = CN(DEXYCB_3D_CONFIG_TEST)
    batch_size = 16
    # train_dataset = DexYCB(cfg_train)
    # train_data = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=16)
    test_dataset = DexYCB(cfg_test)
    test_data = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=16)

    print("Test Dataset Len: ", len(test_dataset))
    print("Test Dataloader Batch: ", len(test_data))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load('model.pt')
    model = model.cuda()

    # hm[-1], heat_map use MSE
    #  so3, so3 use MSE
    #  beta, beta use MSE
    #  joint_root,
    #  bone_vis / 2
    loss_hm = nn.MSELoss()
    loss_so3 = nn.MSELoss()
    #loss_beta = nn.MSELoss()
    loss_joint_root = nn.MSELoss()

    l_hm, l_so3, l_beta, l_joint_root = 1, 1, 1, 1

    lr = 1e-3  # 0.02 is way too large
    epoches = 1
   
    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    testing(test_data, epoches)


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

