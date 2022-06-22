import pathlib

import numpy
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from numpy import *
import scipy
from scipy import signal
import pandas as pd
# 数据读取-单因变量与多因变量
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# 根据分隔符读取数据
def loadDataSet01(filename, Separator=', '):
    fr = open(filename)
    arrayLines = fr.readlines()
    assert len(arrayLines) != 0
    num = len(arrayLines[0].split(Separator)) - 1
    row = len(arrayLines)
    x = mat(zeros((row, num)))
    y = mat(zeros((row, 1)))
    index = 0
    for line in arrayLines:
        curLine = line.strip().split(Separator)
        curLine = [float(i) for i in curLine]
        x[index, :] = curLine[0: -1]
        y[index, :] = curLine[-1]
        index += 1
    return np.array(x), np.array(y)

# 求x,y的 均值与方差
def data_Mean_Std(x0, y0):
    mean_x = mean(x0, 0)
    mean_y = mean(y0, 0)
    std_x = std(x0, axis=0, ddof=1)
    std_y = std(y0, axis=0, ddof=1)
    return mean_x, mean_y, std_x, std_y


# 数据标准化
from sklearn import preprocessing
def stardantDataSet(x0, y0):
    e0 = preprocessing.scale(x0)
    f0 = preprocessing.scale(y0)
    return e0, f0

# 中值滤波
def med_filtering(tea):
    ans = signal.medfilt(tea, 5)
    return ans

# 高斯滤波
def gaussian_filtering(tea):
    ans = scipy.ndimage.filters.gaussian_filter(tea, sigma=0.85, mode="nearest")
    return ans

# 信号的最小二乘平滑 是一种在时域内基于局域多项式最小二乘法拟合的滤波方法。这种滤波器最大的特点在于在滤除噪声的同时可以确保信号的形状、宽度不变。
def savitzky_golay(ans):  # 实现曲线平滑
    return signal.savgol_filter(ans, 7, 3, mode="wrap")

def detrend(ans):
    ans = signal.detrend(ans)  # 去除非线性趋势 去除散射
    return ans
def MSC(data_x):  # 多元散射校正
    ## 计算平均光谱做为标准光谱
    mean = numpy.mean(data_x, axis=0)

    n, p = data_x.shape
    msc_x = numpy.ones((n, p))

    for i in range(n):
        y = data_x[i, :]
        lin = LinearRegression()
        lin.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = lin.coef_  # 线性回归系数
        b = lin.intercept_ # 线性回归截距
        msc_x[i, :] = (y - b) / k
    return msc_x

def EMSC(data_x):  # 多元散射校正
    ## 计算平均光谱做为标准光谱
    mean = numpy.mean(data_x, axis=0)

    n, p = data_x.shape
    msc_x = numpy.ones((n, p))

    for i in range(n):
        y = data_x[i, :]
        lin = LinearRegression()
        lin.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = lin.coef_  # 线性回归系数
        b = lin.intercept_ # 线性回归截距
        msc_x[i, :] = (y - b) / k
    return msc_x
# 信号预处理
def filter(x0):
    # x0 = MSC(x0) #掉0.4%
    xx = np.zeros(shape=(x0.shape[0],x0.shape[1]-2))
    # print(xx.shape)

    for x in range(len(x0)):
        x0[x] = savitzky_golay(x0[x])
        # x0[x] = gaussian_filtering(x0[x])
        x0[x] = detrend(x0[x])

        x0[x] = med_filtering(x0[x])  # 并没有提升
        xx[x] = x0[x][1:-1]
    # x0 = preprocessing.scale(x0) # 标准化

    return xx
    # return x0

import torch
# 获得RR RMSE
RPD_total = 0
def getRR_RMSE(y_test,y_predict,isVal_ = False):
    global RPD_total
    if isinstance(y_test, torch.Tensor):
        row = len(y_test)
        y_mean = torch.mean(y_test, 0).clone().detach()
        SSE = sum(sum(power((y_test.detach().numpy() - y_predict.detach().numpy()), 2), 0))
        SST = sum(sum(power((y_test.detach().numpy() - y_mean.detach().numpy()), 2), 0))
        # SSR = sum(sum(power((y_predict.detach().numpy() - y_mean.detach().numpy()), 2), 0))
        # SSR = SST-SSE
        RR = 1 - SSE / SST
        RMSE = sqrt(SSE / row)
        if isVal_:
            RPD_total += 1 / sqrt(1 - RR)
        return RR, RMSE
    y_mean = mean(y_test, 0)
    row = len(y_test)
    SSE = sum(sum(power((y_test - y_predict), 2), 0))
    SST = sum(sum(power((y_test - y_mean), 2), 0))
    # SSR = sum(sum(power((y_predict - y_mean), 2), 0))
    # print(SSE, SST, SSR)
    RR = 1 - (SSE / SST)
    RMSE = sqrt(SSE / row)
    if isVal_:
        RPD_total += 1/sqrt(1-RR)
    # print("RPD:",1/sqrt(1-RR))
    return RR,RMSE



def PCA(x_train, x_test, y_train, y_test,x_val,y_val,n_components=100):
    from sklearn.decomposition import PCA
    # pls2 = PCA(copy=True,n_components=11, tol=1e-06)
    pls2 = PCA(copy=True,n_components=n_components)
    pls2.fit(x_train)

    x_train = pls2.transform(x_train)
    x_test = pls2.transform(x_test)
    x_val = pls2.transform(x_val)
    from biPls import regressionNet,LS_SVM
    return x_train,x_test,y_train,y_test,x_val,y_val #[41,109]  99 100 159
    # return PLS(x_train,x_test,y_train,y_test,x_val,y_val)  # 102
    # return Linear_Regression(x_train,x_test,y_train,y_test,x_val,y_val) # 102



    # y_predict = pls2.transform(x_test)
    # print(y_predict.shape)
    # print(pls2.components_)
    # print("PCA:" , pls2.explained_variance_ratio_,len(pls2.explained_variance_ratio_))
    # print(pls2.explained_variance_)

    # y_val_ = pls2.predict(x_val)
    # RR,RMSE = getRR_RMSE(y_test,y_predict)
    # RR1,RMSE1 = getRR_RMSE(y_val,y_val_)

    # return RR, RMSE,RR1,RMSE1


def write_to_csv(y_predict,y_test,file_path="./PLS-master/data/Sigmoid_Sigmoid_99.36.pkl"):
    y_predict = torch.reshape(torch.Tensor(y_predict),(-1,1))
    y_test = torch.Tensor(y_test)
    y_predict = torch.concat((y_predict, y_test, y_predict - y_test), dim=1)
    y_predict = pd.DataFrame(y_predict.detach().numpy())
    ff = get_log_name(pre="net", suff="csv", dir_path=str(pathlib.Path(file_path).parent))
    print("csv file save in {}".format(ff))
    y_predict.to_csv(ff)

def LS_SVM(x_train, x_test, y_train, y_test,x_val,y_val):
    from sklearn import svm
    # pls2 = svm.LinearSVR(C=700,tol=1e-6)
    # pls2 = svm.SVR(kernel='rbf',C=1e2,gamma=48)
    # pls2 = svm.SVR(kernel='rbf',C=1000,gamma=48)
    pls2 = svm.SVR(kernel='rbf',C=1000,gamma=48)
    # pls2 = svm.SVR(kernel='rbf',C=10000,gamma=48)
    # pls2 = svm.SVR(kernel='rbf',C=1e5,gamma=2.4)
    # pls2 = svm.SVR(kernel='poly',C=20000000000,degree=2)
    pls2.fit(x_train, y_train.ravel())
    y_predict = pls2.predict(x_test)
    # print("y_predict:" ,y_predict)
    # print("y_test:" , y_test.ravel())
    y_val_ = pls2.predict(x_val)
    RR, RMSE = getRR_RMSE(y_test.ravel(), y_predict)
    RR1, RMSE1 = getRR_RMSE(y_val.ravel(), y_val_,True)

    # write_to_csv(y_val_,y_val)
    # write_to_csv(y_predict,y_test)
    return RR, RMSE, RR1, RMSE1


#  https://blog.csdn.net/qq_41815357/article/details/109637463
def randomForest(x_train, x_test, y_train, y_test, x_val, y_val):
    forest = RandomForestRegressor(
        n_estimators=5,
        min_samples_leaf=10,
        # oob_score=True,
        max_features=20,
        n_jobs=-1)
    forest.fit(x_train, y_train.ravel())
    y_predict = forest.predict(x_test)
    # print("y_predict:" ,y_predict)
    # print("y_test:" , y_test.ravel())
    y_val_ = forest.predict(x_val)
    RR, RMSE = getRR_RMSE(y_test.ravel(), y_predict)
    RR1, RMSE1 = getRR_RMSE(y_val.ravel(), y_val_, True)

    # write_to_csv(y_val_,y_val)
    # write_to_csv(y_predict,y_test)
    return RR, RMSE, RR1, RMSE1

def PCA_randomForest(x_train, x_test, y_train, y_test, x_val, y_val,n_components=31):
    x_train, x_test, y_train, y_test, x_val, y_val = PCA(x_train, x_test, y_train, y_test, x_val, y_val,n_components=n_components)
    return randomForest(x_train, x_test, y_train, y_test, x_val, y_val)

def PCA_LS_SVM(x_train, x_test, y_train, y_test,x_val,y_val,n_components=31):
    x_train, x_test, y_train, y_test, x_val, y_val = PCA(x_train, x_test, y_train, y_test, x_val, y_val,n_components=n_components)
    return LS_SVM(x_train, x_test, y_train, y_test, x_val, y_val)
def Linear_Regression(x_train, x_test, y_train, y_test,x_val,y_val):


    pls2 = LinearRegression()

    pls2.fit(x_train, y_train)
    # print(pls2.coef_)
    # print(pls2.intercept_)
    # print(pls2.score(x_train,y_train))
    y_predict = pls2.predict(x_test)
    y_val_ = pls2.predict(x_val)
    RR, RMSE = getRR_RMSE(y_test, y_predict)
    RR1, RMSE1 = getRR_RMSE(y_val, y_val_,True)
    return RR, RMSE, RR1, RMSE1

def PLS(x_train, x_test, y_train, y_test):
    RR = 0
    RMSE = 0
    start = 11
    num = 1
    for i in range(start, start + num):
        pls2 = PLSRegression(n_components=i, max_iter=750, tol=1e-06, scale=True)
        pls2.fit(x_train, y_train)
        y_predict = pls2.predict(x_test)
        RR,RMSE = getRR_RMSE(y_test,y_predict)
    return RR, RMSE

def PLS(x_train, x_test, y_train, y_test,x_val,y_val):
    RR = 0
    RMSE = 0
    RR1 = 0
    RMSE1 = 0
    start = 11
    num = 1
    for i in range(start, start + num):
        pls2 = PLSRegression(n_components=i, max_iter=750, tol=1e-06, scale=True)
        pls2.fit(x_train, y_train)
        y_predict = pls2.predict(x_test)
        y_val_ = pls2.predict(x_val)
        RR, RMSE = getRR_RMSE(y_test, y_predict)
        RR1, RMSE1 = getRR_RMSE(y_val, y_val_,True)
    return RR, RMSE, RR1, RMSE1




def split10items(x0, y0, splitss=10, random_state=1, extend=1):
    import random
    random.seed(random_state)
    len = np.shape(x0)[0]
    a = list(np.arange(0, len))
    random.shuffle(a)
    u = 0
    r = int(len / splitss)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(splitss - 1):
        x_test.append(x0[a[u:u + r]])
        y_test.append(y0[a[u:u + r]])

        train_x = list(a[u + r:])
        train_x.extend(a[0:u])

        if extend > 1:
            p = list(train_x)
            for i in range(1, extend):
                train_x.extend(p)
        x_train.append(x0[train_x])
        y_train.append(y0[train_x])
        u += r

    x_test.append(x0[a[u:]])
    y_test.append(y0[a[u:]])
    x_train.append(x0[a[0:u]])
    y_train.append(y0[a[0:u]])
    return x_train, x_test, y_train, y_test


def Cross_validation(x0, y0, f_test, splits=10, random_state=11, extend=1):
    x0 = filter(x0)
    x_trains, x_tests, y_trains, y_tests = split10items(x0, y0, splits=splits, random_state=random_state, extend=extend)
    p = 0
    m = 0
    for i in range(len(x_trains)):
        a, b = f_test(x_trains[i], x_tests[i], y_trains[i], y_tests[i])
        p += a
        m += b

    print(u"R^2 {0}%".format(np.round(p / len(x_trains) * 100, 2)))
    print(u"RMSE.", m / len(x_trains))
j = 0

def get_log_name(pre="recode",suff= "log",dir_path="./"):
    import pathlib
    import re
    kk = re.compile("(\d+)")
    o = [str(i.stem) for i in pathlib.Path(dir_path).glob("{}*.{}".format(pre,suff))]
    import datetime
    day = str(datetime.date.today())
    day = day[day.index('-')+1:].replace("-","_")
    max1 = 0
    for po in o:
        u = re.search(kk, po)
        if u != None:
            m = int(u.group(0))
            max1 = max(m, max1)
    return "{}/{}{}_{}.{}".format(pathlib.Path(dir_path),pre,max1 + 1,day,suff)
def getSplitsAndIndices(split_len=10):
    l = np.shape(x0)[1]
    len1 = int(np.floor(l / split_len))  # 剩余的尾巴不要了
    a = list(np.arange(l))
    b = list(np.arange(len1))
    splits = []
    u = 0
    for i in range(len1):
        splits.append(a[u:u + split_len])
        u += split_len
    # splits.append(a[u:])
    return splits, b
from sklearn.model_selection import train_test_split
def main1(s_len = 11):
    import time
    start = time.time()
    global x0,bb,splits

    x0, y0 = loadDataSet01('./PLS-master/data/test_all_reflect1.txt', ', ')  # 单因变量与多因变量
    x0 = filter(x0)


    splits, bb = getSplitsAndIndices(split_len=s_len)

    # print(getNext(1))
    # print(getNext(2))
    # print(getNext(3))
    # print(getNext(23))
    m = 0
    m_j = 0
    b_ = []

    rm = 10
    rm_j = 1
    mylog = open(get_log_name(), mode='a', encoding='utf-8')

    while len(bb) > ceil(11.0 / s_len):
        k = 0
        p = get_iter(x0)
        max = 0
        max_j = 0
        for x in p:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y0, test_size=0.2, random_state=3)

            RR, RMSE = PLS(x_train, x_test, y_train, y_test)
            # print("{} RR: {} RMSE: {}".format(k, RR, RMSE), file=mylog)
            if max < RR:
                max = RR
                max_j = k
            if rm > RMSE:
                rm = RMSE
                rm_j = len(bb)
            k += 1

        if m < max or abs(max-m) < 0.001:

            m_j = len(bb)
            b_ = list(bb)
            if m < max:
                m = max
        print("max_RR: {}, delete group is {}".format(max,bb[max_j]), file=mylog)
        bb.remove(bb[max_j])
        print(bb, file=mylog)

    print(file=mylog)
    print('the best groups: {}'.format(b_), file=mylog)
    print("R2_max:{}, b_len: {}".format(m, m_j), file=mylog)
    print("rmse_min: {}, b_len: {}".format(rm,rm_j),file=mylog)
    # print(bb,file=mylog)
    end = time.time()
    print("the spent time is {} seconds".format((end - start)),file=mylog)
    mylog.close()
    # print(x)
    # print(np.shape(x))
    # break
    # print(len)
    # print(x0)
def get_iter(x0):
    global bb

    for i in range(len(bb)):
        f = list(bb[0:i])
        f.extend(bb[i + 1:])  # index
        ans = list(splits[f[0]])
        for v in f[1:]:
            ans.extend(splits[v])  # splits
        xx = []
        for m in x0:
            xx.append(m[ans])
        yield np.array(xx)  # x0


def getDataIndex(x0, index):
    l1 = np.shape(x0)[0]
    mm = []
    for i in range(l1):
        mm.append(np.array(x0[i][index]))
    return np.array(mm)

if __name__=='__main__':
    main1()