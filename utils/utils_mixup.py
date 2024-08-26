from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def mix_data_lab(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    index = torch.randperm(batch_size)

    lam = max(lam, 1-lam)
    mixed_x = lam * x + (1-lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, index, lam

