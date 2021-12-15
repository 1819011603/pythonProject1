import torch

a = torch.tensor(3., requires_grad=True)
b = torch.tensor(4., requires_grad=True)

print(a.requires_grad)

f1 = 2 * a
f2 = a * b
z = f1 + f2
print(z)
z.backward()
print(f1.grad)
print(f2.grad)
print(a.grad)
print(b.grad)

with torch.no_grad():  # 不算grad
    f3 = a * b
    # True
    print(f2.requires_grad)
    # False
    print(f3.requires_grad)

a1 = a.detach()  # 把a1从计算图中分离出来 不会计算a1的grad
print(f'a1.requires_grad={a1.requires_grad}')
print(f'a1.requires_grad={a.requires_grad}')
