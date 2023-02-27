import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as Datasets
import torchvision.transforms as transforms

from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self, z_dim, g_hidden, image_channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, g_hidden*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_hidden*8),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(g_hidden*8, g_hidden*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden*4),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(g_hidden*4, g_hidden*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(g_hidden*2, g_hidden, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(g_hidden, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            )
    def forward(self, x):
        return self.main(x)
    
class Discriminator(nn.Module):
    def __init__(self, image_channels, d_hidden):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(image_channels, d_hidden, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(d_hidden, d_hidden*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_hidden*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_hidden*2, d_hidden*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_hidden*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_hidden*4, d_hidden*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_hidden*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_hidden*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x).view(-1,1).squeeze(1)
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    image_channels = 3
    z_dim = 100
    g_hidden = 64
    x_dim = 64
    d_hidden = 64
    epochs = 25
    REAL = 1.
    FAKE = 0.
    lr = 2e-4

    transform = transforms.Compose([
        transforms.Resize(x_dim),
        transforms.CenterCrop(x_dim),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = Datasets.CelebA(root="./data", transform=transform, download=True)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    G = Generator(z_dim, g_hidden, image_channels).to(device)
    D = Discriminator(image_channels, d_hidden).to(device)

    print(G)
    print(D)

    criterion = nn.BCELoss()

    optim_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
    for epoch in range(epochs):
        iterator = tqdm(train_loader)
        for img, _ in iterator:
            x_real = img.to(device)

            real_label = torch.full((x_real.size(0),), REAL, device=device)
            fake_label = torch.full((x_real.size(0),), FAKE, device=device)

            D.zero_grad()
            y_real = D(x_real)
            loss_D_real = criterion(y_real, real_label)
            loss_D_real.backward()

            z = torch.randn(x_real.size(0), z_dim, 1, 1, device=device)
            x_fake = G(z)
            y_fake = D(x_fake.detach())
            loss_D_fake = criterion(y_fake, fake_label)
            loss_D_fake.backward()
            optim_D.step()

            G.zero_grad()
            y_fake_r = D(x_fake)
            loss_G = criterion(y_fake_r, real_label)
            loss_G.backward()
            optim_G.step()

            iterator.set_description("G Loss : {}, D Loss real: {}, D Loss fake : {}".format(loss_G.item(), loss_D_real.item(), loss_D_fake.item()))

    torch.save(G.state_dict(), "./model/G_dcgan.pth")
    torch.save(D.state_dict(), "./model/D_dcgan.pth")
