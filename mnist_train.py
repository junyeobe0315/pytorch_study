import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data.dataset import Dataset

import torchvision.datasets as datasets
import torchvision.transforms as Transforms

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=Transforms.ToTensor()
)

class Denoising(Dataset):
    def __init__(self):
        self.mnist = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=Transforms.ToTensor()
        )
        for i in range(len(self.mnist)):
            