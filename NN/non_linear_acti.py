import torch
import torch.nn as nn

input = torch.tensor([[1, -0.5],
                      [-0.1, 3.6]])

input = torch.reshape(input, (-1, 1, 2, 2))
# print(input.size())


class Test(nn.Module):

    def __init__(self):
        super(Test, self).__init__()

        # self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):

        # x = self.relu(x)
        x = self.sig(x)

        return x

test = Test()

out = test(input)

print(out)



