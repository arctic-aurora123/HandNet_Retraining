import argparse
import torch
import numpy as np


def get_intr(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cx = 323.52
    cy = 237.07
    fx = 600.568
    fy = 600.69

    while True:
        if img is None:
            continue
        if img.shape[1] > img.shape[2]:
            margin = int((img.shape[1] - img.shape[2]) / 2)
            cy = cy - margin
            width = img.shape[2]
        elif img.shape[1] < img.shape[2]:
            margin = int((img.shape[2] - img.shape[1]) / 2)
            cx = cx - margin
            width = img.shape[1]
        cx = (cx * 256)/width
        cy = (cy * 256)/width
        fx = (fx * 256)/width
        fy = (fy * 256)/width
        break

    intr = torch.from_numpy(np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)).unsqueeze(0).to(device)
    return intr
