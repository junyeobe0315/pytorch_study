import torch
import torch.nn as nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        modules = []
        for dim in hidden_dims:
            modules.append(
                nn.Sequential(
                nn.Conv2d(in_channels, out_channels=dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dim),
                nn.LeakyReLU()
                )
            )
            in_channels = dim
        self.encoder = nn.Sequential(*modules)
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.var = nn.Linear(hidden_dims[-1], latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)
        hidden_dims.reverse()

        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i],
                                   hidden_dims[i + 1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.LeakyReLU()
                )
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1, stride=2),
            nn.Tanh()
        )

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.mu(result)
        var = self.var(result)
        return [mu, var]
    
    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        return self.final_layer(result)
    
    def reparameterize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x):
        mu, var = self.encode(x)
        z = self.reparameterize(mu, var)
        return [self.decode(z), x, mu, var]
    
    def loss(self, x_hat, x, mu, var, **kwargs):

        kld_weight = kwargs['M_N']
        recons_loss =F.mse_loss(x_hat, x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
        
    def sample(self, num_samples, device, **kwargs):
        z = torch.randn(num_samples).to(device)
        return self.decode(z)
    
    def generate(self, x, **kwargs):
        return self.forward(x)[0]
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.datasets import CelebA
    from torchvision import transforms
    import torch.optim as optim
    from tqdm import tqdm

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CelebA(root='./data', transform=transform, download=True, split="train")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    in_channels = 3
    latent_dim = 32*32
    hidden_dims = [32, 64, 128, 256, 512]
    vae = VAE(in_channels, latent_dim, hidden_dims).to(device)
    lr = 0.001
    betas = (0.5, 0.999)
    optimizer = optim.Adam(vae.parameters(), lr=lr, betas=betas)

    epochs = 100
    for epoch in range(epochs):
        iterator = tqdm(train_loader)
        for img, _ in iterator:
            img = img.to(device)
            batch_size = img.shape[0]

            vae.zero_grad()
            [x_hat, x, mu, var] = vae(img)
            loss = vae.loss(x_hat, x, mu, var, M_N=0.005)
            loss["loss"].backward()
            optimizer.step()

            iterator.set_description("epoch ; {}, loss : {}".format(epoch+1, loss['loss']))
    torch.save(vae.state_dict(), "./model/vae.pth")