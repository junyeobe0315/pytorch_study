from vae import VAE
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA, CIFAR10
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

transform = transforms.Compose([
transforms.ToTensor(),
])

test_dataset = CIFAR10(root='./data', transform=transform, download=True, train=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

in_channels = 3
latent_dim = 32*32
hidden_dims = [32, 64, 128, 256, 512]
vae = VAE(in_channels, latent_dim, hidden_dims)

vae.load_state_dict(torch.load("./model/vae.pth"))
vae.eval()
with torch.no_grad():
    plt.figure(figsize=(8,8))
    for i in range(64):
        sample = vae.sample(1, torch.device("cpu"))
        plt.subplot(8, 8, i+1)
        plt.imshow(sample.numpy())
    plt.show()