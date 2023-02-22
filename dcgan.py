import torch
import torch.nn as nn


class G_Basic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(G_Basic, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, bias=False,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)
    

class D_Basic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(D_Basic, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, 
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, input_size, hidden_dims):
        super(Generator, self).__init__()
        self.first = nn.Sequential(
            nn.ConvTranspose2d(input_size, hidden_dims[0], 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU()
        )
        self.block = nn.Sequential(
            G_Basic(hidden_dims[0], hidden_dims[1], 4, 2, 1),
            G_Basic(hidden_dims[1], hidden_dims[2], 4, 2, 1),
            G_Basic(hidden_dims[2], hidden_dims[3], 4, 2, 1),
            nn.ConvTranspose2d(hidden_dims[3], 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.first(x)
        x = self.block(x)
        return x
    

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dims):
        super(Discriminator, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(input_size, hidden_dims[0], 4, 2, 1, bias=False),
            nn.ReLU()
        )
        self.block = nn.Sequential(
            D_Basic(hidden_dims[0], hidden_dims[1], 4, 2, 1),
            D_Basic(hidden_dims[1], hidden_dims[2], 4, 2, 1),
            D_Basic(hidden_dims[2], hidden_dims[3], 4, 2, 1),
            nn.Conv2d(hidden_dims[3], 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.first(x)
        print(x.shape)
        x = self.block(x)
        return x
    


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.datasets import CelebA, CIFAR10
    from torchvision import transforms
    import torch.optim as optim
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = CIFAR10(root="./data", train=True, transform=trans)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    input_size = 100

    hidden_dims = [64*8, 64*4, 64*2, 64]
    G = Generator(input_size=input_size, hidden_dims=hidden_dims).to(device)
    hidden_dims.reverse()
    D = Discriminator(input_size=3, hidden_dims=hidden_dims).to(device)
    criterion = nn.BCELoss()

    optimizerD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    print(G)
    print(D)

    epochs = 5
    for epoch in range(epochs):
        iterator = tqdm(train_loader)
        for img, _ in iterator:
            D.zero_grad()
            
            real = img[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), 1, dtype=torch.float, device=device)
            output = D(real).view(-1)
            errD_ = criterion(output, label)
            errD_.backward()

            z = torch.randn(32, 100, 1, 1).to(device)
            fake = G.forward(z)
            label.fill_(0.)
            output = D(fake.detach()).view(-1)
            errD = criterion(output, label)
            errD.backward()
            optimizerD.step()
            errD += errD_

            G.zero_grad()
            label.fill_(1.)
            output = D(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            iterator.set_description("epoch : {}, D Loss : {}, G Loss : {}".format(epoch+1, errD.item(), errG.item()))

