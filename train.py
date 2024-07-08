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
from manolayer import ManoLayer

from torch.utils.tensorboard import SummaryWriter


datapath = '/mnt/public/datasets'
img_w = 256
img_h = 256

writer = SummaryWriter('runs/0704')

# LOAD the data
# dict_keys(['rot_rad', 'rot_mat3d', 'affine', 'image', 'target_bbox_center', 'target_bbox_scale', 'target_joints_2d', 'target_joints_vis', 
# 'image_path', 'affine_postrot', 'target_cam_intr', 'target_joints_3d', 'target_verts_3d', 'target_mano_pose',
# 'target_mano_shape', 'idx', 'image_full', 'cam_center', 'bbox_center', 'bbox_scale', 'cam_intr', 'joints_2d', 'joints_3d', 
# 'verts_3d', 'joints_vis', 'joints_uvd', 'verts_uvd', 'bone_scale', 'mano_pose', 'mano_shape', 'raw_size'])
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

def get_hm_taget_new(pred_hm, joints_2d):
    # by TTT at 2024.4.22
    # TODO: We could remove the for-loop by using advanced indexing techniques,
    #       but we should do it later.
    b, c, h, w = pred_hm.shape
    target_hm = np.zeros((b, c, h, w))
    # print(target_hm.shape)  # B, 21, 64, 64
    # print(joints_2d.shape)  # B, 21, 2
    joints_2d = torch.tensor(joints_2d / 4, dtype=torch.long)
    for batch_idx in range(b):
        for channel_idx in range(c):
            x, y = joints_2d[batch_idx][channel_idx]
            if 0 <= x < w and 0 <= y < h:
                target_hm[batch_idx][channel_idx][y][x] = 1
                target_hm[batch_idx][channel_idx] = cv2.GaussianBlur(target_hm[batch_idx][channel_idx],(5,5),0)
                #plt.imshow(target_hm[batch_idx][channel_idx])
                #plt.show()
            # else: pass - do not draw on the image
    target_hm = torch.tensor(target_hm, dtype=torch.float32)
    #print(target_hm)  # B, 21, 64, 64
    return target_hm


def training(train_data, test_data, epoches):
    n_iter = 0
    for n_epochs in range(epoches):
        model.train()
        with tqdm(total=len(train_data)) as pbar1:
            pbar1.set_description(f'Epoch {n_epochs} Training: ')
            
            for idx, data in enumerate(train_data):
                imgs = data['image'].to(device)
                #so3_target = torch.tensor(data['target_mano_pose'], device=device)
                #joint_root_target = torch.tensor(data['target_joints_3d'][:,0:1,:], device=device)
                
                joint_3d_target = torch.tensor(data['target_joints_3d'],device=device)
                verts_3d_target = torch.tensor(data['target_verts_3d'], device=device)
                intr = torch.tensor(data['target_cam_intr'], device=device)

                optimizer.zero_grad()
                
                hm, so3, beta, _ , _ , joints_3d = model(imgs, intr)
                so3 = so3.unsqueeze(1).detach().cpu()
                beta = beta.detach().cpu()
                verts_3d, _ , _ = mano_layer(th_pose_coeffs = so3, th_betas = beta)
                verts_3d = verts_3d.to(device)
                # hm_target = hm_graph(batch_size, data['joints_2d']).to(device)
                hm_target = get_hm_taget_new(hm, data['target_joints_2d']).to(device)
                
                #so3 = torch.split(so3, 16, dim=1)
                #so3 = torch.stack(so3, dim=2)
                
                # print("target_joints_2d shape: {}".format(hm_target.shape))
                # print("joint_3d_target shape: {}".format(joint_3d_target.shape))
                # print("verts_3d_target shape: {}".format(verts_3d_target.shape))
                # print("joints_2d shape: {}".format(hm.shape))
                # print("joints_3d shape: {}".format(joints_3d.shape))   
                # print("verts_3d shape: {}".format(verts_3d.shape))
                
                loss_hm_value = loss_hm(hm_target, hm) * l_hm
                loss_3d_value = loss_3d(joint_3d_target, joints_3d) * l_3d
                loss_mano_value = loss_mano(verts_3d_target, verts_3d) * l_mano
                
                #loss_so3_value = loss_so3(so3_target, so3)
                #loss_joint_root_value = loss_hm(joint_root_target, joint_root)
                #loss = l_hm * loss_hm_value + l_so3 * loss_so3_value + l_joint_root * loss_joint_root_value
                loss = loss_hm_value + loss_3d_value + loss_mano_value
                
                loss.backward()
                optimizer.step()
                # print("Train loss: {}".format(loss))
                n_iter += 1
                if n_iter % 10 == 0:
                    writer.add_scalar('Train_Loss/2d', loss_hm_value.item(), n_iter)
                    writer.add_scalar('Train_Loss/3d', loss_3d_value.item(), n_iter)
                    writer.add_scalar('Train_Loss/mano', loss_mano_value.item(), n_iter)
                    writer.add_scalar('Train_Loss/SUM', loss.item(), n_iter)
                pbar1.update(1)

        with tqdm(total=len(test_data)) as pbar2:
            pbar2.set_description(f'Epoch {n_epochs} Testing: ')
            # test
            with torch.no_grad():
                model.eval()
                for data in test_data:
                    imgs = data['image'].to(device)
                    #so3_target = torch.tensor(data['target_mano_pose'], device=device)
                    #joint_root_target = torch.tensor(data['target_joints_3d'][:,0:1,:], device=device)
                    joint_3d_target = torch.tensor(data['target_joints_3d'], device=device)
                    verts_3d_target = torch.tensor(data['target_verts_3d'], device=device)
                    intr = torch.tensor(data['target_cam_intr'], device=device)
                    
                    hm, so3, beta, _ , _, joints_3d = model(imgs, intr)
                    so3 = so3.unsqueeze(1).detach().cpu()
                    beta = beta.detach().cpu()
                    verts_3d, _ , _ = mano_layer(th_pose_coeffs = so3, th_betas = beta)
                    verts_3d = verts_3d.to(device)
                    # hm_target = hm_graph(batch_size, data['joints_2d']).to(device)
                    hm_target = get_hm_taget_new(hm, data['target_joints_2d']).to(device)

                    loss_hm_value = loss_hm(hm_target, hm) * l_hm
                    loss_3d_value = loss_3d(joint_3d_target, joints_3d) * l_3d
                    loss_mano_value = loss_mano(verts_3d_target, verts_3d) * l_mano
                    
                    #loss_so3_value = loss_so3(so3_target, so3)
                    #loss_joint_root_value = loss_hm(joint_root_target, joint_root)
                    #loss = l_hm * loss_hm_value + l_so3 * loss_so3_value + l_joint_root * loss_joint_root_value
                    loss = loss_hm_value + loss_3d_value + loss_mano_value
                    #print("Test loss: {}".format(test_loss))
                    pbar2.update(1)
                    
                #update the scalar
                writer.add_scalar('Test_Loss/2d', loss_hm_value.item(), n_epochs)
                writer.add_scalar('Test_Loss/3d', loss_3d_value.item(), n_epochs)
                writer.add_scalar('Test_Loss/mano', loss_mano_value.item(), n_epochs)
                writer.add_scalar('Test_Loss/SUM', loss.item(), n_epochs)
                    
        # save weights
        print('-----saving the model and the weights-----')
        torch.save(model, f'model_0704_add3D.pt')
        torch.save(model.state_dict(), 'weights_0704_add3D.pt')
        

if __name__ == '__main__':
    cfg_train = CN(DEXYCB_3D_CONFIG_TRAIN)
    cfg_test = CN(DEXYCB_3D_CONFIG_TEST)
    batch_size = 64
    train_dataset = DexYCB(cfg_train)
    train_data = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=16)
    test_dataset = DexYCB(cfg_test)
    test_data = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=16)


    print("Train Dataset Len: ", len(train_dataset))
    print("Test Dataset Len: ", len(test_dataset))
    print("Train Dataloader Batch: ", len(train_data))
    print("Test Dataloader Batch: ", len(test_data))
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HandNet()
    model = model.cuda()

    mano_layer = ManoLayer(center_idx=9, side="right", mano_root="./assets/models", use_pca=False, flat_hand_mean=True)
    # hm[-1], heat_map use MSE
    #  so3, so3 use MSE
    #  beta, beta use MSE
    #  joint_root,
    #  bone_vis / 2
    loss_hm = nn.MSELoss()
    loss_mano = nn.MSELoss()
    loss_3d = nn.MSELoss()
    #loss_so3 = nn.MSELoss()
    #loss_beta = nn.MSELoss()
    #loss_joint_root = nn.MSELoss()

    l_hm, l_mano, l_3d = 1.0, 0.0001, 1.0

    lr = 1e-4  # 0.02 is way too large
    epoches = 100
   
    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    data = next(iter(train_data))
    
    training(train_data, test_data, epoches)
    #training(data, data, epoches)
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


# def hm_graph(batch_size, joints_2d):
#     #define the resize shape
#     hm_target_batch = list()
#     w = 64
#     h = 64
#     for i in range(batch_size):
#         hm_target = list()
#         joints_2d /= 4
#         for j in range(joints_2d.shape[1]):
            
#             img = np.zeros((w, h), dtype=np.int32)  #(H, W)
#             center = joints_2d[i, j, :].numpy().astype(np.int32).tolist()
#             print(center)
#             for c in center:
#                 # print("c: ".format(c))
#                 img[c[0]][c[1]] = 1
#             heat_map = torch.tensor(img).unsqueeze(0)
#             #resize, the interpolation mode is NEAREST, may be others
            
#             # to visualize , can be deleted
#             # heat_map = heat_map.permute(1,2,0)
#             # graph = heat_map.numpy().squeeze(2).copy()
#             # print(graph.shape)
#             # cv2.imwrite('./joint_image/joint_image{}.jpg'.format(j), graph)
#             print(img)
#             hm_target.append(heat_map)
#         #heat_map = heat_map.permute(1,2,0) #(H, W, C), to visualize , can be deleted
        
#         hm_stack = torch.stack(hm_target)
#         hm_target_batch.append(hm_stack)
        
#         # used to visualize , can be deleted
#         # img = heat_map.numpy().copy()  
#         # print(heat_map.shape)
#         # cv2.imwrite('./joint_image/joint_image{}.jpg'.format(i), img)
#     hm_target_batch = torch.stack(hm_target_batch).squeeze(2).squeeze(3).type(torch.float32)
#     return hm_target_batch
