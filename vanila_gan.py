import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    input_size = 100
    hidden_size = 128
    output_size = 28*28

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(input_size, hidden_size, output_size)
    discriminator = Discriminator(output_size, hidden_size)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    criterion = nn.BCELoss()
    lr = 0.0002
    betas = (0.5, 0.999)
    generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    epochs = 100
    for epoch in range(epochs):
        iterator = tqdm(train_loader)
        for real_images, _ in iterator:
            real_images = real_images.to(device)
            batch_size = real_images.shape[0]

            # Train discriminator with real images
            discriminator.zero_grad()
            real_labels = torch.ones(batch_size, 1)
            real_labels = real_labels.to(device)
            real_output = discriminator(real_images.view(batch_size, -1))
            real_loss = criterion(real_output, real_labels)
            real_loss.backward()

            # Train discriminator with generated images
            noise = torch.randn(batch_size, input_size, device=device)
            fake_images = generator(noise)
            fake_labels = torch.zeros(batch_size, 1)
            fake_labels = fake_labels.to(device)
            fake_output = discriminator(fake_images.detach().view(batch_size, -1))
            fake_loss = criterion(fake_output, fake_labels)
            fake_loss.backward()
            discriminator_optimizer.step()

            # Train generator with updated discriminator
            generator.zero_grad()
            noise = torch.randn(batch_size, input_size, device=device)
            fake_images = generator(noise)
            fake_labels = torch.ones(batch_size, 1).to(device)
            fake_output = discriminator(fake_images.view(batch_size, -1))
            generator_loss = criterion(fake_output, fake_labels)
            generator_loss.backward()
            generator_optimizer.step()

            iterator.set_description("epoch : {}, G loss : {}, D loss : {}".format(epoch+1, generator_loss.item(), real_loss.item()))
    torch.save(generator.state_dict(), "./model/G.pth")
    torch.save(discriminator.state_dict(), "./model/D.pth")