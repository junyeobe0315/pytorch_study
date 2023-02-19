import matplotlib.pyplot as plt
from vanila_gan import Generator
import torch



input_size = 100
hidden_size = 128
output_size = 28*28

generator = Generator(input_size, hidden_size, output_size)
generator.load_state_dict(torch.load("./model/G.pth"))
generator.eval()
with torch.no_grad():
    noise = torch.randn(64, input_size)
    images = generator(noise).view(64, 1, 28, 28)
    images = (images + 1) / 2
    plt.figure(figsize=(8, 8))
    for i in range(64):
        plt.subplot(8, 8, i+1)
        plt.imshow(images[i].squeeze().numpy(), cmap='gray')
        plt.axis('off')
    plt.show()