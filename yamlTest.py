import yaml


s = ['welcome', 'to', 'changsha']
t = {'help':'city',"hello":'word'}


print(s)
print(t)


print(yaml.dump(s))
print(yaml.dump(t))


import torch
N, C, H, W = 2, 3, 4, 5
images = torch.zeros(2, 3, 4, 5)
print(images.stride()) # (60, 20, 5, 1)      NCHW   print(images.is_contiguous()) ## True   ，is_contiguous默认是contiguous_format image是连续存储的返回true
images_cl = images.contiguous(memory_format=torch.channels_last)
print(images_cl.is_contiguous()) ## False
print(images_cl.is_contiguous(memory_format=torch.contiguous_format)) ## True
u = torch.zeros_like(images_cl,memory_format=torch.channels_last)
print(u.stride())
print(u.is_contiguous())
print(u.is_contiguous(memory_format=torch.channels_last))
print(torch.zeros_like(images_cl,memory_format=torch.channels_last))