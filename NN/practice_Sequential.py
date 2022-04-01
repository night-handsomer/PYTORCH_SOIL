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
                       batch_size=64)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Conv1
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32,
                               kernel_size=(5, 5),
                               stride=1,
                               padding=2
                               )
        # maxpool
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # Conv2
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=(5, 5),
                               stride=1,
                               padding=2
                               )
        # maxpool
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # Conv3
        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=(5, 5),
                               stride=1,
                               padding=2
                               )
        # maxpool3
        self.maxpool3 = nn.MaxPool2d((2, 2))

        # flatten
        self.flatten = nn.Flatten()

        # linear
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(64, 10)


        """
            demo2:Sequential的使用
        """
        self.model1 = nn.Sequential(

            # conv1, maxpool
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),

            # conv2, maxpool
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),

            # conv3, maxpool
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),

            # flatten
            nn.Flatten(),

            # linear
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )



    def forward(self, x):

        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = self.maxpool3(self.conv3(x))
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


        """
            demo2:Sequential的使用
        """
        x = self.model1(x)
        return x


net = Net()

print(net)


# 网络检测
input = torch.ones(size=(64, 3, 32, 32))
out = net(input)
print(out.shape)
