import numpy
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from numpy import *
import scipy
from scipy import signal
import pandas as pd
# 数据读取-单因变量与多因变量
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

# 信号预处理
def filter(x0):
    # x0 = MSC(x0)
    # x0 = D1(x0)
    # x0 = preprocessing.scale(x0) # 标准化
    for x in range(len(x0)):
        x0[x] = savitzky_golay(x0[x])
        # x0[x] = gaussian_filtering(x0[x])
        x0[x] = detrend(x0[x])
        x0[x] = med_filtering(x0[x])  # 提升0.2%
    return x0


# 获得RR RMSE
def getRR_RMSE(y_test,y_predict):
    if isinstance(y_test, torch.Tensor):
        row = len(y_test)
        y_mean = torch.tensor(mean(y_test, 0), dtype=torch.float64)
        SSE = sum(sum(power((y_test.detach().numpy() - y_predict.detach().numpy()), 2), 0))
        SST = sum(sum(power((y_test.detach().numpy() - y_mean.detach().numpy()), 2), 0))
        SSR = sum(sum(power((y_predict.detach().numpy() - y_mean.detach().numpy()), 2), 0))
        # SSR = SST-SSE
        RR = 1 - SSE / SST
        RMSE = sqrt(SSE / row)
        return RR, RMSE
    y_mean = mean(y_test, 0)
    row = len(y_test)
    SSE = sum(sum(power((y_test - y_predict), 2), 0))
    SST = sum(sum(power((y_test - y_mean), 2), 0))
    # SSR = sum(sum(power((y_predict - y_mean), 2), 0))
    # print(SSE, SST, SSR)
    RR = 1 - (SSE / SST)
    RMSE = sqrt(SSE / row)
    return RR,RMSE


def PLS(x_train, x_test, y_train, y_test):
    RR = 0
    RMSE = 0
    start = 11
    num = 1
    for i in range(start, start + num):
        pls2 = PLSRegression(n_components=i, max_iter=1000, tol=1e-03, scale=True)
        pls2.fit(x_train, y_train)
        y_predict = pls2.predict(x_test)
        RR,RMSE = getRR_RMSE(y_test,y_predict)
    return RR, RMSE


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
import torch
from  RegressionNet import Regression
import torch.optim as optim
def regressionNet(x_train, x_test, y_train, y_test):
    net = Regression()
    global j
    print(net)
    y_mean = torch.tensor(mean(y_test, 0), dtype=torch.float64)
    x_train = torch.tensor(x_train, dtype=torch.float64)
    x_test = torch.tensor(x_test, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64)
    y_test = torch.tensor(y_test, dtype=torch.float64)

    row = len(y_test)
    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.937, 0.999))
    loss_func = torch.nn.MSELoss()

    for i in range(30000):
        y_predict = net(x_train)
        loss = loss_func(y_predict, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(y_predict)

    with torch.no_grad():
        y_predict = net(x_test)
        # print(list(y_predict.detach().numpy() - y_test.detach().numpy()))
        SSE = sum(sum(power((y_test.detach().numpy() - y_predict.detach().numpy()), 2), 0))
        SST = sum(sum(power((y_test.detach().numpy() - y_mean.detach().numpy()), 2), 0))
        SSR = sum(sum(power((y_predict.detach().numpy() - y_mean.detach().numpy()), 2), 0))
        # SSR = SST-SSE
        RR = 1 - SSE / SST
        """
        RMSE实际上描述的是一种离散程度，不是绝对误差，其实就像是你射击，你开10枪，我也开10枪，你和我的总环数都是90环，你的子弹都落在离靶心差不多距离的地方,
        而我有几枪非常准，可是另外几枪又偏的有些离谱，这样的话我的RMSE就比你大，反映出你射击的稳定性高于我，但在仅以环数论胜负的射击比赛中，我们俩仍然是平手。
        这样说你应该能理解吧，基本可以理解为稳定性。那么对于你预报来说，在一个场中RMSE小说明你的模式对于所有区域的预报水平都相当，反之，RMSE较大，
        那么你的模式在不同区域的预报水平存在着较大的差异。

        """

        RMSE = sqrt(SSE / row)
        j += 1
        s = ["L"]
        for i, item in enumerate(net.l.named_children()):

            if i % 2 == 1:
                s.append(str(item[1]).split('(')[0])
            else:
                s.append(str(item[1]).split("in_features=")[1].split(",")[0])
        print(round(RR * 100, 2), RMSE)
        torch.save(net.state_dict(), "./{0}_{1}.pkl".format("_".join(s),round(RR*100,2)))
    return RR, RMSE
def get_log_name(dir_path="./"):
    import pathlib
    import re
    kk = re.compile("(\d+)")
    o = [str(i.stem) for i in pathlib.Path(dir_path).glob("recode*.log")]

    max1 = 0
    for po in o:
        u = re.search(kk, po)
        if u != None:
            m = int(u.group(0))
            max1 = max(m, max1)
    return "recode{}.log".format(max1 + 1)
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