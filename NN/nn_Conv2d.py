import torch
import torchvision
import torch.nn as nn


from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 使用 transform 将数据集的数据转化为张量
trans_data = transforms.Compose(transforms=[
    transforms.ToTensor()
])

# 准备数据集，原始格式为 PIL 的
train = torchvision.datasets.CIFAR10(root="../trans/dataset3", train=True, transform=trans_data, download=True)

# 使用 DataLoader 来装载数据

train_set = DataLoader(dataset=train,
                       batch_size=64,
                       shuffle=True
                       )


# 定义网络模型
class OneNet(nn.Module):
    """
        由于后续的reshape问题，这里只给出一层卷积，但是本身的网络定义（带注释的那些）是没错的。
    """
    def __init__(self):
        super(OneNet, self).__init__()
        # conv1
        self.Conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=0
                               )
        # conv2
        # self.Conv2 = nn.Conv2d(in_channels=10,
        #                        out_channels=10,
        #                        kernel_size=(3, 3),
        #                        padding=0
        #                        )
        # self.relu = nn.ReLU()

    def forward(self, x):

        x = self.Conv1(x)
        # x = self.relu(self.Conv1(x))

        # return self.relu(self.Conv2(x))
        return x


one_net = OneNet()
# print(one_net)

writer = SummaryWriter("co")

step = 0
for data in train_set:
    imgs, labels = data
    output = one_net(imgs)          # 将图片输入到神经网络中
    # print(type(output))
    # tensor.size ---> ([64, 10, 28, 28])  这个是先查看一下输出的格式
    # print(output.size())
    output = torch.reshape(output, (-1, 3, 30, 30))   # 这里只为查看输出结果。这个写法是不严谨的，如果遇到二零非线性激活和多一层卷积的话，就会报错
    writer.add_images("test", output, step)
    step += 1











