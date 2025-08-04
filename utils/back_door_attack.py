
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

def apply_backdoor_attack(dataset, trigger_label=0, poison_fraction=0.01, patch_size=3):
    num_poison = int(len(dataset) * poison_fraction)
    indices = torch.randperm(len(dataset))[:num_poison]

    for idx in indices:
        image, _ = dataset[idx]
        image[0, 0:patch_size, 0:patch_size] = 1.0  # white patch top-left
        dataset.data[idx] = (image * 255).byte()
        dataset.targets[idx] = trigger_label

    return dataset
    
