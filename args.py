import argparse
from turtle import width
import torch
import numpy as np

def get_intr():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cx = 323.52
    cy = 237.07
    fx = 600.568
    fy = 600.69
    
    width = 256
    cx = (cx * 256)/width
    cy = (cy * 256)/width
    fx = (fx * 256)/width
    fy = (fy * 256)/width

    intr = torch.from_numpy(np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)).unsqueeze(0).to(device)
    return intr
    
