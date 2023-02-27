from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as Transforms
import torch.utils.data as data
import torch.nn as nn

from resnet import ResNet
from mobilenet_v1 import *

transforms = Transforms.Compose([
    Transforms.RandomCrop((32,32), padding=4),
    Transforms.RandomHorizontalFlip(p=0.5),
    Transforms.ToTensor(),
    Transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    Transforms.Resize(224)
])

train_data = torchvision.datasets.cifar.CIFAR10(root="./data", train=True, download=True, transform=transforms)

train_loader = data.DataLoader(train_data, batch_size=64, shuffle=True)

if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("device : ", device)

model = MobileNet(in_channels=3, num_classes=10)
model.to(device)

lr = 1e-4
optim = torch.optim.Adam(model.parameters(), lr=lr)

for epochs in range(30):
    iterator = tqdm(train_loader)
    for data, label in iterator:
        optim.zero_grad()
        preds = model(data.to(device))

        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

        iterator.set_description("epoch : {} / loss : {}".format(epochs+1, loss.item()))

torch.save(model.state_dict(), "./model/MobileNet_v1.pth")