

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets.samples_generator import make_blobs
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本3个特征，共4个簇
X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3,3, 3], [0,0,0], [1,1,1], [2,2,2]], cluster_std=[0.2, 0.1, 0.2, 0.2],
                  random_state =9)
print(X.shape)
print(y.shape)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(X,y)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
pca = PCA(n_components=2)
pca.fit(X,y)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

X_new = pca.transform(X)
print(X_new.shape)
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVR
# from sklearn.metrics import r2_score
#
# np.random.seed(0)
#
# x = np.random.randn(80, 20)
# y = x[:, 0] + 2 * x[:, 1] + np.random.randn(80)
#
# clf = SVR(kernel='linear', C=1.25)
# x_tran, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# clf.fit(x_tran, y_train)
# y_hat = clf.predict(x_test)
#
# print("得分:", r2_score(y_test, y_hat))
#
# r = len(x_test) + 1
# print(y_test)
# plt.plot(np.arange(1, r), y_hat, 'go-', label="predict")
# plt.plot(np.arange(1, r), y_test, 'co-', label="real")
# plt.legend()
# plt.show()