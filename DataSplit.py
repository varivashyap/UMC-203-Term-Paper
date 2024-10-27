import os
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset
from PIL import Image
import numpy as np
from dataclasses import dataclass
import copy


def train_test_divide(train, transform):
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    all_data = trainset + testset

    train_size = int(0.2 * len(all_data))

    if train:
        return Subset(all_data, range(0, train_size))
    else:
        return Subset(all_data, range(train_size, len(all_data)))
    
    