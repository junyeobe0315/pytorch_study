import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import torchvision.datasets as datasets
import torchvision.transforms as Transforms

from noise import gaussian_noise
from auto_encoder import *

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
        self.data = []
        for i in range(len(self.mnist)):
            noise_input = gaussian_noise(self.mnist.data[i])
            input_tensor = torch.tensor(noise_input)
            self.data.append(torch.unsqueeze(input_tensor, dim=0))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.mnist.data[idx] / 255
        return data, label

if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

trainset = Denoising()
train_loader = DataLoader(trainset, batch_size=32)

model = AutoEncoder().to(device)

lr = 0.001
optim = torch.optim.Adam(params=model.parameters(), lr=lr)

for epoch in range(20):
    iterator = tqdm(train_loader)
    for data, label in iterator:
        optim.zero_grad()
        pred = model(data.to(device))

        loss = nn.MSELoss()(torch.squeeze(pred), label.to(device))
        loss.backward()
        optim.step()
        iterator.set_description("epoch : {}, loss : {}".format(epoch+1, loss.item()))

torch.save(model.state_dict(), "./model.pth")