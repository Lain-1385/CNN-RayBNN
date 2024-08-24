import numpy as np
import raybnn_python
import torch 
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
import os
from torchvision import datasets, transforms,utils
from torch.utils.data import ConcatDataset, Subset, DataLoader
from torch import optim
import torch.nn.functional as F

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = ConcatDataset([
        datasets.CIFAR10(root="/home/lain1385/scratch/project/data_tao", transform=transform, train=True, download=True),
        datasets.CIFAR10(root="/home/lain1385/scratch/project/data_tao", transform=transform, train=False, download=True)
    ])

    return 1

if __name__ == "__main__":
    flag = main()