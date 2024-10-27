import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from networks import Encoder

def train_test_divide(train, transform):
    data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    Data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


    data = np.array(data.data)
    Data = np.array(Data.data)

    # concatenate data and Data
    data_array = np.concatenate((data, Data), axis=0)

    if train:
        # train_array is 1st 20% of data_array
        train_set = data_array[:int(len(data_array)*0.1)]
        train_set = torch.utils.data.DataLoader(train_set)
        return train_set
    else:
        # test_array is remaining 80% of data_array
        test_set = data_array[int(len(data_array)*0.1):]
        test_set = torch.utils.data.DataLoader(test_set)
        return test_set
    
