import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)

        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x_ = x
        
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)

        x_ = self.downsample(x_)
        x += x_
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()

        self.block1 = BasicBlock(in_channels=3, out_channels=64)
        self.block2 = BasicBlock(in_channels=64, out_channels=128)
        self.block3 = BasicBlock(in_channels=128, out_channels=256)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.full_connect1 = nn.Linear(in_features=256*4*4, out_features=2048)
        self.full_connect2 = nn.Linear(in_features=2048, out_features=512)
        self.full_connect3 = nn.Linear(in_features=512, out_features=num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.full_connect1(x)
        x = self.relu(x)
        x = self.full_connect2(x)
        x = self.relu(x)
        x = self.full_connect3(x)
        
        return x
