from vae import VAE
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))
])

test_dataset = CelebA(root='./data', transform=transform, download=True, split="test")
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

in_channels = 3
latent_dim = 32*32
hidden_dims = [32, 64, 128, 256, 512]
vae = VAE(in_channels, latent_dim, hidden_dims)

vae.load_state_dict(torch.load("./model/vae.pth"))
vae.eval()
with torch.no_grad():
    plt.figure(figsize=(8,8))
    for i in range(64):
        z = torch.randn(1024)
        sample = vae.decode(z)[0].reshape(32,32,3)
        plt.subplot(8, 8, i+1)
        plt.imshow(sample.numpy())
    plt.show()