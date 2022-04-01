import torch

from torch import nn





class Test(nn.Module):
    
    def __init__(self):
        super(Test, self).__init__()


    def forward(self, x):

        output = x + 1
        return output


test = Test()

x = torch.tensor([1, 2.])
output = test(x)
print(output)


