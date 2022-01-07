import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

class Regression(nn.Module):
    def __init__(self,c = 256):
        '''
        pls
            R^2 96.31%
            RMSE. 2.3259371956037143
        beyesi
            R^2 96.52%
            RMSE. 2.2561172126769575


        nn.Linear(256,32),
        nn.ELU(),
        nn.Linear(32,11),
        nn.ELU(),
        nn.Linear(11,1)
            R^2 96.95%
            RMSE. 2.1626438290203582

        5000 epochs: R^2 98.09%  RMSE. 1.72267558623318 R^2 98.02%  RMSE. 1.7117793067374287

        50000 epochs: R^2 99.88%  R^2 99.51%  RMSE. 0.8098495115100194  R^2 100.22% RMSE. 0.86005852112458

        100000 epochs: R^2 97.91% RMSE. 1.6634818023228792

        nn.Linear(256,32),
        nn.Mish(),
        nn.Linear(32,11),
        nn.Mish(),
        nn.Linear(11,1)

        5000 epochs R^2 98.19% RMSE. 1.602569202394975
        50000epochs 99.87% RMSE. 0.6585095016979408
        100000epochs R^2 99.63%RMSE. 0.33076927820012125  R^2 99.87% RMSE. 0.4080463246936687
        '''
        super(Regression,self).__init__()
        self.l = nn.Sequential(

            nn.Linear(c,32,dtype=torch.float64), # 35 12  91.97   32 11
            nn.Sigmoid(),

            nn.Linear(32,11,dtype=torch.float64),
            nn.Sigmoid(),
            # nn.Conv1d(c, c, 5, 1, 2),
            nn.Linear(11,1,dtype=torch.float64)
        )
    def forward(self,x):
        # x = torch.unsqueeze(x,dim=2)
        return self.l(x)

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()
        self.rnn = nn.Sequential(
            nn.LSTM(32,10,dtype=torch.float64)
        )
        self.l = nn.Sequential(
            nn.Linear(80, 17, dtype=torch.float64),  # 35 12  91.97   32 11
            nn.Sigmoid(),
            nn.Linear(17, 1, dtype=torch.float64)
        )
        self.h0 = torch.randn(2,416,10)
        self.c0 = torch.randn(2,416,10)


    def forward(self, x):
        x= x.reshape(-1,8,32)
        x = x.permute(1,0,2)

        y = self.rnn(x)
        x = y[0]
        x = x.permute(1,0,2).reshape(-1,80)


        return self.l(x)
class Transformer(nn.Module):
   def __init__(self):
       super(Transformer, self).__init__()

if __name__ == "__main__":
    net = Regression()
    for i,item in enumerate(net.l.named_children()):
        if i%2==1:
            print(str(item[1])[:-2])

"""


nn.Linear(256,32),
nn.LeakyReLU(),
nn.Linear(32,11),
nn.LeakyReLU(),
nn.Linear(11,1)
R^2 96.69%
RMSE. 2.210449886150209

leakyRelu 
R^2 88.18%
RMSE. 4.464364657322802

nn.Linear(256,32),
nn.Sigmoid(),
nn.Linear(32,11),
nn.LeakyReLU(),
nn.Linear(11,1)
R^2 92.84%
RMSE. 2.949128187208566


"""
# net = Regression()
# print(net)

# optimizer=optim.SGD(net.parameters(),lr=0.001)
# loss_func = torch.nn.MSELoss()
# for i in range(200):
#     predition = net(x)
#     loss = loss_func(predition,y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

