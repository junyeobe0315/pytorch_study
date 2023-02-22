import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import random
from pandas_datareader import data
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import quandl

import matplotlib.pyplot as plt
import seaborn as sns

result_path = Path('./TimeGAN')
if not result_path.exists():
    result_path.mkdir()
hdf_store = result_path / 'timegan.pth'

seq_len = 24
n_seq = 4
batch_size = 128
hidden_dim = 24
num_layers = 3
train_step = 1000
gamma = 1

api_key = 'oWYyVMdphZuSHk84TYBA'
start_date = "1990-01-01"
end_date = '2022-12-30'

tickers = ['AAPL', 'IBM', 'MSFT', 'WMT']
data = pd.DataFrame(columns=tickers)
for ticker in tickers:
    data[ticker] = quandl.get("WIKI/"+ticker, start_date=start_date, end_date=end_date, api_key=api_key)['Adj. Close']
 
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data).astype(np.float32)

dataset = []
for i in range(len(data) - seq_len):
    dataset.append(scaled_data[i:i+seq_len])

n_windows = len(dataset)

class TimeSeries(Dataset):
    def __init__(self, data, random=False, seq_len=24, n_seq=4):
        self.data = data
        self.random = random
        self.seq_len = seq_len
        self.n_seq = n_seq

    def __len__(self):
        return len(self.data)
    
    def make_random_data(seq_len, n_seq):
        while True:
            yield torch.tensor(np.random.uniform(low=0, high=1, size=(seq_len, n_seq)))

    def __getitem__(self, idx):
        if self.random:
            return self.make_random_data(self.seq_len, self.n_seq)
        else:
            return torch.tensor(data[idx])
    
random_train_dataset = DataLoader(TimeSeries(data, random=True, seq_len=seq_len, n_seq=n_seq))
true_train_dataset = DataLoader(TimeSeries(data, random=False, seq_len=seq_len, n_seq=n_seq))

class Time_GAN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, activation=torch.sigmoid, rnn_type="gru"):
        super(Time_GAN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.rnn_type = rnn_type

        if self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)
        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        if self.rnn_type in ["rnn", "gru"]:
            hidden = self.init_hidden(batch_size)
        if self.rnn_type == "lstm":
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).float()
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).float()
            hidden = (h0, c0)

        out, hidden = self.rnn(hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        if self.sigma == nn.Identity:
            idendity = nn.Identity
            return idendity(out)
        out = self.sigma(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
    
lr = 0.001
embedder = Time_GAN(input_size=seq_len*batch_size, output_size=hidden_dim, hidden_dim=hidden_dim, n_layers=3)
embedder_optim = optim.Adam(embedder.parameters(), lr=lr)
recovery = Time_GAN(input_size=hidden_dim, output_size=n_seq, hidden_dim=hidden_dim, n_layers=3)
recovery_optim = optim.Adam(recovery.parameters(), lr=lr)

generator = Time_GAN(input_size=seq_len*batch_size, output_size=hidden_dim, hidden_dim=hidden_dim, n_layers=3)
generator_optim = optim.Adam(generator.parameters(), lr=lr)
discriminator = Time_GAN(input_size=hidden_dim, output_size=1, hidden_dim=hidden_dim, n_layers=3)
discriminator_optim = optim.Adam(discriminator.parameters(), lr=lr)

superviser = Time_GAN(input_size=hidden_dim, output_size=hidden_dim, hidden_dim=hidden_dim, n_layers=2)
superviser_optim = optim.Adam(superviser.parameters(), lr=lr)

class Autoencoder(nn.Module):
    def __init__(self, embedder, recovery):
        super(Autoencoder, self).__init__()
        self.embedder = embedder
        self.recovery = recovery
    def forward(self, x):
        x = self.embedder(x)
        x = self.recovery(x)
        return x

autoencoder = Autoencoder(embedder, recovery)

mse = nn.MSELoss()
bce = nn.BCELoss()

print(embedder)
print(recovery)
print(generator)
print(discriminator)
print(superviser)