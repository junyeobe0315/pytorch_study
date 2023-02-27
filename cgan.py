import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Generator(nn.Module):
    def __init__(self, classes, channels, img_size, latent_dim):
        super(Generator, self).__init__()
        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.label_embedding = nn.Embedding(self.classes, self.classes)
        self.adv_loss = torch.nn.BCELoss()
        
        self.model = nn.Sequential(
            *self.create_layer(self.latent_dim + self.classes, 128, False),
            *self.create_layer(128, 256),
            *self.create_layer(256, 512),
            *self.create_layer(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )
    
    def create_layer(self, size_in, size_out, normalize=True):
        layers = [nn.Linear(size_in, size_out)]
        if normalize:
            layers.append(nn.BatchNorm1d(size_out))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers
    
    def forward(self, z, label):
        z = torch.cat((self.label_embedding(label), z), -1)
        x = self.model(z)
        x = x.view(x.size(0), *self.img_shape)
        return x
    
    def loss(self, output, label):
        return self.adv_loss(output, label)

class Discriminator(nn.Module):
    def __init__(self, classes, channels, img_size, latent_dim):
        super(Discriminator, self).__init__()
        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self. latent_dim = latent_dim
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.label_embedding = nn.Embedding(self.classes, self.classes)
        self.adv_loss = torch.nn.BCELoss()
        self.model = nn.Sequential(
            *self.create_layer(self.classes + int(np.prod(self.img_shape)), 1024, False, True),
            *self.create_layer(1024, 512, True, True),
            *self.create_layer(512, 256, True, True),
            *self.create_layer(256, 128, False, False),
            *self.create_layer(128, 1, False, False),
            nn.Sigmoid()
        )

    def create_layer(self, size_in, size_out, drop_out=True, act_func=True):
        layers = [nn.Linear(size_in, size_out)]
        if drop_out:
            layers.append(nn.Dropout(0.5))
        if act_func:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers
    
    def forward(self, image, label):
        x = torch.cat((image.view(image.size(0), -1), self.label_embedding(label)), -1)
        return self.model(x)
    
    def loss(self, output, label):
        return self.adv_loss(output, label)
    

class MODEL(object):
    def __init__(self, name, device, data_loader, classes, channels, img_size, latent_dim):
        self.name = name
        self.device = device
        self.data_loader = data_loader
        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        
        if self.name == "cgan":
            self.D = Discriminator(self.classes, self.channels, self.img_size, self.latent_dim)
            self.D.to(self.device)
            self.G = Generator(self.classes, self.channels, self.img_size, self.latent_dim)
            self.G.to(self.device)
        
        self.optim_G = None
        self.optim_D = None

    def create_optim(self, lr, alpha=0.5, beta=0.999):
        self.optim_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), 
                                        lr=lr, betas=(alpha, beta))
        
        self.optim_D = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), 
                                        lr=lr, betas=(alpha, beta))
        
    def train(self, epochs, log_interval=100, out_dir="./model", verbose=True):
        self.G.train()
        self.D.train()
        
        for epoch in range(epochs):
            iterator = tqdm(self.data_loader)
            for data, label in iterator:
                data, label = data.to(self.device), label.to(self.device)
                batch_size = data.size(0)
                real_label = torch.full((batch_size, 1), 1., device=self.device)
                fake_label = torch.full((batch_size, 1), 0., device=self.device)

                self.G.zero_grad()
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                x_fake_labels = torch.randint(0, self.classes, (batch_size,), device=self.device)
                x_fake = self.G(z, x_fake_labels)
                y_fake_g = self.D(x_fake, x_fake_labels)
                g_loss = self.G.loss(y_fake_g, real_label)
                g_loss.backward()
                self.optim_G.step()

                self.D.zero_grad()
                y_real = self.D(data, label)
                d_real_loss = self.D.loss(y_real, real_label)

                y_fake_d = self.D(x_fake.detach(), x_fake_labels)
                d_fake_loss = self.D.loss(y_fake_d, fake_label)
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.optim_D.step()

                iterator.set_description("epoch : {}, G Loss : {}, D Loss {}".format(epoch+1, g_loss.item(), d_loss.item()))


def main(train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    if train:
        dataset = datasets.MNIST(root="./data", download=True, transform=transform)
        data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
        model = MODEL("cgan", device=device, data_loader=data_loader, classes=10, channels=1, img_size=64, latent_dim=100)
        model.create_optim(lr=0.0002)

        model.train(epochs=200, log_interval=100, out_dir="./model", verbose=False)
        model.save_to('./model')


if __name__ == "__main__":
    main(train=True)