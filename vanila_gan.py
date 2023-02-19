import torch
import torch.nn as nn
import torchvision.datasets as Datasets
import torchvision.transforms as Transforms
import torch.utils.data as data
import torchvision

import glob
import imageio

import matplotlib.pyplot as plt
import numpy as np
import os

import PIL
from PIL import Image
import time
import random

train_data = Datasets.CIFAR10(root="./data", download=True, train=True)
test_data = Datasets.CIFAR10(root="./data", download=True, train=False)

class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

transforms = Transforms.Compose([
    Transforms.RandomHorizontalFlip(p=0.5),
    Transforms.Resize(32,32,3),
    Transforms.ToTensor()
])


class Train(Datasets):
    def __init__(self):
        self.cifar = Datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms
        )
        def __len__(self):
            return len(self.cifar)
        def __getitem__(self, idx):
            data, label = self.cifar[idx]
            data /= 255
            return data, label

class Test(Datasets):
    def __init__(self):
        self.cifar = Datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transforms
        )
        def __len__(self):
            return len(self.cifar)
        def __getitem__(self, idx):
            data, label = self.cifar[idx]
            data /= 255
            return data, label

if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

trainset = Train()
testset = Test()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 4*4*256, bias=False),
            nn.LeakyReLU(),
            nn.
        )