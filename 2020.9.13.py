import torch


x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)

z = y*y*3
out = z.mean()
print(z, out)

a = torch.randn(2,2)
a = ((a * 3) /(a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
b.requires_grad_(True)
print(b.grad_fn)

print(b.backward())

# x.grad 的值为 None，因为 x 没有将 requires_grad 设为 True
x = torch.tensor(3.)
w = torch.tensor(4.,requires_grad=True)
b = torch.tensor(5.,requires_grad=True)

y = w * x + b
print(y)
print(y.backward())
print(x.grad)
print(w.grad)
print(b.grad)







