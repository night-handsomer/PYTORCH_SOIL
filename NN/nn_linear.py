import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms


dataset = torchvision.datasets.CIFAR10(root='../trans/dataset3',
                                       train=True,
                                       transform=transforms.ToTensor(),
                                       download=True
                                       )

train_set = DataLoader(dataset=dataset,
                       batch_size=64,
                       shuffle=True
                       )


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.linear = nn.Linear(196608, 10)

    def forward(self, x):

        x = self.linear(x)

        return x


net = Net()


for data in train_set:
    imgs, labels = data
    output = torch.flatten(imgs)
    out = net(output)
    print(out.shape)


