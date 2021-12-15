import numpy as np
import torch


arr = np.ones((3,3))
print(type(arr))
print(arr.dtype)
t = torch.tensor(arr)
print(t.dtype)
print(t)


arr = np.array([[1,3,43],[2,23,1]])
print(type(arr))
print(arr.dtype)
print(arr)
t = torch.tensor(arr,dtype=torch.int64) # 深拷贝
print(t)
arr[1:] = 0  # 修改np.ndarray 不会改变tensor的值
print(arr)
print(t)


arr = np.array([[1,3,43],[2,23,1]])
print(type(arr))
print(arr.dtype)
print(arr)
t = torch.from_numpy(arr) # 浅拷贝
print(t)
arr[1:] = 0  # 修改np.ndarray 会改变tensor的值
print(arr)
print(t)

out_t = torch.tensor([1])
t = torch.zeros((3,3),out=out_t)
print(t,"\n",out_t)
print(id(t),id(out_t),id(t) == id(out_t))