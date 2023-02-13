import torchvision
import torchvision.transforms as Transforms
import torch.utils.data as data
import torch
from resnet import ResNet
from mobilenet_v1 import MobileNet

transforms = Transforms.Compose([
    Transforms.RandomCrop((32,32), padding=4),
    Transforms.RandomHorizontalFlip(p=0.5),
    Transforms.ToTensor(),
    Transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    Transforms.Resize(224)
])

if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"

test_data = torchvision.datasets.cifar.CIFAR10(root="./data", train=False, download=True, transform=transforms)
test_loader = data.DataLoader(test_data, batch_size=32, shuffle=False)

model = MobileNet(in_channels=3)
model.load_state_dict(torch.load("./model/MobileNet_v1.pth"))
model.to(device)
model.eval()
num_corr = 0

with torch.no_grad():
    for data, label in test_loader:
        preds = model(data.to(device))
        preds = preds.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr

acc = num_corr / len(test_data)    
print("Accuracy : {}".format(acc))