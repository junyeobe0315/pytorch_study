from auto_encoder import *
import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as Transforms
from noise import gaussian_noise
import matplotlib.pyplot as plt


model = AutoEncoder()
model.cpu()
if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

test_data = MNIST(
    root="./data",
    download=True,
    train=False,
    transform=Transforms.ToTensor()
)

with torch.no_grad():
    model.load_state_dict(torch.load("./model.pth", map_location=device))
    img = test_data.data[0]
    gaussian = gaussian_noise(img)

    input_ = torch.unsqueeze(gaussian, dim=0)
    input_.type(torch.FloatTensor)
    input_.to(device)
    input_ = torch.unsqueeze(input_, dim=0)

    plt.subplot(1,3,1)
    plt.imshow(torch.squeeze(gaussian))
    plt.subplot(1,3,2)
    plt.imshow(torch.squeeze(model(input_)))
    plt.subplot(1,3,3)
    plt.imshow(torch.squeeze(img))
    plt.show()