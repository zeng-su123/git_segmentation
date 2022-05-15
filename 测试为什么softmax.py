import torch
import  torch.nn as nn
import torch.nn.functional as F

x = torch.randn([3,2,2,2])
y = F.softmax(x,dim=1)
print(x)
print(y)
