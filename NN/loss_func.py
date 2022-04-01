import torch
import torchvision
import torch.nn as nn

from torch.utils.data import DataLoader


dataset = torchvision.datasets.CIFAR10(root='../trans/dataset3',
                                       train=True,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)


train_set = DataLoader(dataset=dataset,
                       batch_size=64)


class Test(nn.Module):

    def __init__(self):
        super(Test, self).__init__()
        # 网络
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

        x = self.model1(x)
        return x


net = Test()

# 网络输出测试
# input = torch.ones(size=(64, 3, 32, 32))
# print(net(input))

loss = nn.CrossEntropyLoss()

for data in train_set:
    imgs, labels = data
    output = net(imgs)
    result = loss(output, labels)
    result.backward()
    print("ok")


