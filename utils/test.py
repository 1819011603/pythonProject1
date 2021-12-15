#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/23 22:09
# @Author  : hejindong
# @File    : asd.py
# @Software: PyCharm


import numpy as np
import matplotlib.pyplot as plt
import cv2

def myImageFilter(img0, h):#卷积操作
    #填充图像
    c=h.shape[0]//2
    k=h.shape[1]//2
    img=np.pad(img0, (c, k), mode="reflect")
    #计算for循环的范围
    c=img.shape[0]-h.shape[0]+1
    k=img.shape[1]-h.shape[1]+1
    dc=h.shape[0]
    dk=h.shape[1]
    ans=np.zeros((c,k))
    #卷积计算
    for i in range(c):
        for j in range(k):
            ans[i,j]=np.sum(np.multiply(img[i:i+dc, j:j+dk], h))
    return ans

image=cv2.imread("image2.jpg")
print(image)
image=image[:,:,(2,1,0)]#由BGR转RGB
plt.imshow(image)
plt.show()
image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)#转灰度图
plt.imshow(image, cmap="gray")
plt.show()

kernel=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image_gary=myImageFilter(image, kernel)
plt.imshow(image_gary, cmap="gray")
plt.show()