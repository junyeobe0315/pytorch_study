import torch
import torch.nn as nn

class Depth_wise(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Depth_wise, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                stride=stride, groups=in_channels, bias=False, padding=padding)
        self.batchnorm1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        return x

class Batchnorm_std(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Batchnorm_std, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    
class MobileNet(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super(MobileNet, self).__init__()
        self.bn1 = Batchnorm_std(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.dw2 = Depth_wise(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dw3 = Depth_wise(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.dw4 = Depth_wise(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dw5 = Depth_wise(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.dw6 = Depth_wise(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dw7 = Depth_wise(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.dw8 = Depth_wise(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.dw9 = Depth_wise(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.dw10 = Depth_wise(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.dw11 = Depth_wise(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.dw12 = Depth_wise(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.dw13 = Depth_wise(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0)
        self.dw14 = Depth_wise(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.bn1(x)
        x = self.dw2(x)
        x = self.dw3(x)
        x = self.dw4(x)
        x = self.dw5(x)
        x = self.dw6(x)
        x = self.dw7(x)
        x = self.dw8(x)
        x = self.dw9(x)
        x = self.dw10(x)
        x = self.dw11(x)
        x = self.dw12(x)
        x = self.dw13(x)
        x = self.dw14(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x