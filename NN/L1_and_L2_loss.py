import torch
import torch.nn as nn


# 使用L1损失函数之前，要先实例化。
# nn.L1Loss(reduction="mean")  这里的 reduction 参数默认是mean，
# 意思是说对应点差的绝对值相加后再取平均；也可以是sum，即只是相加。
loss = nn.L1Loss()

input = torch.tensor([[1, 1.1, 0.9]])

label = torch.ones(size=(1, 3))

result = loss(input, label)
print(result)

