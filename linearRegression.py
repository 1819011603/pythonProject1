import numpy as np
import torch

np.random.seed(0)

X = np.array([[1, 2], [2, 3], [4, 6], [3, 1]], dtype=np.float)
y = np.array([[8], [13], [26], [9]], dtype=np.float)
# y = 2 * x1 + 3 * x2
w = np.random.rand(2, 1)
iter_count = 20  # 迭代次数
lr = 0.02


# return 4×1
def forward(x):
    return np.matmul(x, w)


def loss(y, y_pred):
    return ((y - y_pred) ** 2 / 2).sum()


criterion = torch.nn.MSELoss(reduction="sum")


def gradient(x, y, y_pred):
    return np.matmul(x.T, y_pred - y)


for i in range(iter_count):
    y_pred = forward(X)
    l = loss(y, y_pred)
    print(f'iter {i}, loss {l}')
    grad = gradient(X, y, y_pred)
    w -= lr * grad

print(f'final parameter: {w}')

torch.manual_seed(0)

X = torch.from_numpy(np.array([[1, 2], [2, 3], [4, 6], [3, 1]], dtype=np.float))
y = torch.from_numpy(np.array([[8], [13], [26], [9]], dtype=np.float))


def forward1(x):
    return torch.matmul(x, w)
lr = 0.01

w = torch.randn(2, 1, requires_grad=True, dtype=torch.double)
for i in range(iter_count):
    y_pred = forward1(X)
    l = criterion(y,y_pred)
    print(f'iter {i}, loss {l}')
    l.backward()
    # grad = gradient(X, y, y_pred)
    with torch.no_grad():
        w -= lr * w.grad
        w.grad.zero_()

print(f'final parameter: {w}')

optimizer = torch.optim.SGD([w,],lr)

for i in range(iter_count):
    y_pred = forward1(X)
    l = criterion(y,y_pred)
    print(f'iter {i}, loss {l}')
    l.backward()
    # grad = gradient(X, y, y_pred)
    optimizer.step()
    optimizer.zero_grad()

print(f'final parameter: {w}')

print("myModel")

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand(2,1,dtype=torch.double))

    def forward(self,x):
        return torch.matmul(x,self.w)


model = MyModel()
optimizer = torch.optim.SGD(model.parameters(),lr)

for i in range(iter_count):
    y_pred =model.forward(X)
    l = criterion(y,y_pred)
    print(f'iter {i}, loss {l}')
    l.backward()
    # grad = gradient(X, y, y_pred)
    optimizer.step()
    optimizer.zero_grad()

print(f'final parameter: {model.w}')