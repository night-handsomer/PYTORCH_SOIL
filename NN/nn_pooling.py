import torch
import torch.nn as nn


img = torch.tensor([[1, 2, 0, 3, 1],
                    [0, 1, 2, 3, 1],
                    [1, 2, 1, 0, 0],
                    [5, 2, 3, 1, 1],
                    [2, 1, 0, 1, 1]], dtype=torch.float32)

img = torch.reshape(img, shape=(1, 1, 5, 5))

class Test(nn.Module):

    def __init__(self):
        super(Test, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        x = self.maxpool(x)
        return x

net = Test()

output = net(img)
print(output)














