import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
def apply_label_flipping(dataset, flip_rate=0.1, seed=42):
    torch.manual_seed(seed)
    n_samples = len(dataset)
    n_flip = int(n_samples * flip_rate)
    flip_indices = torch.randperm(n_samples)[:n_flip]

    for idx in flip_indices:
        original = dataset.targets[idx].item()
        new_label = torch.randint(0, 10, (1,)).item()
        while new_label == original:
            new_label = torch.randint(0, 10, (1,)).item()
        dataset.targets[idx] = new_label

    return dataset
    
