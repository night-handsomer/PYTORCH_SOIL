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

# 设置优化器
optim = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(5):
    for img, labels in train_set:
        epoch_loss = 0.00
        # 梯度清零
        optim.zero_grad()
        # 将数据输入到网络中
        output = net(img)
        # 计算交叉熵损失（就是计算误差）
        result = loss(output, labels)
        # 计算每一轮的总损失
        epoch_loss += result
        print("第{}次的损失为：{}".format(epoch, epoch_loss))
        # 反向传播
        result.backward()
        # 自动参数更新
        optim.step()


