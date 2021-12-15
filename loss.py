import numpy as np
import torch

def loss(y ,y_pred):
    return ((y-y_pred) ** 2).sum()


criterion = torch.nn.MSELoss(reduction='sum')

y1 = torch.randn(4,1,dtype=torch.float)
y2 = torch.randn(4,1,dtype = torch.float)


# 两个function相同
print(loss(y1,y2))
print(criterion(y1,y2))