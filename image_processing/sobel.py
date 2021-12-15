import matplotlib.pyplot as plt
import numpy
import cv2
import numpy as np
import pylab

img = cv2.imread("img1.png",0)
# r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

k = numpy.array([
    [1, 0, -1], [2, 0, -2], [1, 0, -1]
])/2/ (2**0.5)
k2 = numpy.array([
    [1, 2, 1], [0, 0, 0], [-1, -2, -1]
])/2/ (2**0.5)
res = np.array(cv2.filter2D(img, -1, k),dtype=np.int32)
res1 = np.array(cv2.filter2D(img, -1, k2),dtype=np.int32)
res2 = numpy.floor((res**2 + res1 **2)**0.5)
print(res2)
res2 = np.where(res2 > 40,40,res2)
res2 = numpy.array(res2,numpy.uint8)
# print(res2)
# plt.imshow(res, cmap ='gray')
# plt.plot()
# pylab.show()
# plt.imshow(res1, cmap ='gray')
# plt.plot()
# pylab.show()
print(res2.shape)
plt.imshow(res2, cmap ='gray')
plt.plot()
pylab.show()
