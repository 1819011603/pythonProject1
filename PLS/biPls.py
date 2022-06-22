import numpy as np
from sklearn.model_selection import train_test_split

from utils import *


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


def getNext(j):
    global splits, bb
    bb.remove(j)
    if len(bb) == 0:
        return None
    ans = list(splits[bb[0]])
    for f in bb[1:]:
        ans.extend(splits[f])
    return ans


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


import pathlib


def testRR(index=None, file_path="./PLS-master/data/Sigmoid_Sigmoid_99.36.pkl"):
    if index is None:
        index = list(np.arange(256))
    x0, y0 = loadDataSet01('./PLS-master/data/test.txt', ', ')  # 单因变量与多因变量
    x0 = filter(x0)
    x0 = getDataIndex(x0, index)
    net = Regression(len(index))
    pre = torch.load(file_path)

    net.load_state_dict(pre, strict=False)
    with torch.no_grad():
        y_predict = net(torch.as_tensor(torch.from_numpy(x0), dtype=torch.float64))

        # a = torch.round(torch.as_tensor(y_predict, dtype=torch.float64)) # 四舍五入
        a = torch.as_tensor(y_predict, dtype=torch.float64)  #
        b = torch.tensor(y0, dtype=torch.float64)

        y_predict = torch.concat((a, b, a - b), dim=1)
        RR, RMSE = getRR_RMSE(a.detach().numpy(), y0)
        y_predict = pd.DataFrame(y_predict.detach().numpy())
        ff = get_log_name(pre="net", suff="csv", dir_path=str(pathlib.Path(file_path).parent))
        print("csv file save in {}".format(ff))
        y_predict.to_csv(ff)
        print("RR:", RR)
        print("RMSE:", RMSE)


import torch
from RegressionNet import Regression
import torch.optim as optim


def regressionNet(x_train, x_test, y_train, y_test):
    global j, index
    net = Regression(len(index))
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
        # SSR = sum(sum(power((y_predict.detach().numpy() - y_mean.detach().numpy()), 2), 0))
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
        torch.save(net.state_dict(), "./{0}_{1}.pkl".format("_".join(s), round(RR * 100, 2)))
    return RR, RMSE


def regressionNet(x_train, x_test, y_train, y_test, x_val, y_val):
    global j, index
    net = Regression(len(x_train[0]))
    # print(net)

    x_train = torch.tensor(x_train, dtype=torch.float64)
    x_test = torch.tensor(x_test, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64)
    y_test = torch.tensor(y_test, dtype=torch.float64)

    x_val = torch.tensor(x_val, dtype=torch.float64)
    y_val = torch.tensor(y_val, dtype=torch.float64)

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
        y_val_ = net(x_val)


        """
        RMSE实际上描述的是一种离散程度，不是绝对误差，其实就像是你射击，你开10枪，我也开10枪，你和我的总环数都是90环，你的子弹都落在离靶心差不多距离的地方,
        而我有几枪非常准，可是另外几枪又偏的有些离谱，这样的话我的RMSE就比你大，反映出你射击的稳定性高于我，但在仅以环数论胜负的射击比赛中，我们俩仍然是平手。
        这样说你应该能理解吧，基本可以理解为稳定性。那么对于你预报来说，在一个场中RMSE小说明你的模式对于所有区域的预报水平都相当，反之，RMSE较大，
        那么你的模式在不同区域的预报水平存在着较大的差异。

        """


        RR, RMSE = getRR_RMSE(y_test,y_predict)
        RR1,RMSE1 = getRR_RMSE(y_val,y_val_,True)

        j += 1
        s = ["L"]
        for i, item in enumerate(net.l.named_children()):

            if i % 2 == 1:
                s.append(str(item[1]).split('(')[0])
            else:
                s.append(str(item[1]).split("in_features=")[1].split(",")[0])
        # print(round(RR * 100, 2), RMSE, round(RR1 * 100, 2), RMSE1)
        # torch.save(net.state_dict(), "./{0}_{1}.pkl".format("_".join(s), round(RR * 100, 2)))
    return RR, RMSE, RR1, RMSE1


def main(f_test=PLS, s_len=11, splitss=10, random_state=11):
    import time
    start = time.time()
    global x0, bb, splits

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
    rm_j = 0
    mylog = open(get_log_name(), mode='a', encoding='utf-8')

    while len(bb) > ceil(11.0 / s_len):
        k = 0
        p = get_iter(x0)
        max = 0
        max_j = 0
        rmse = 10
        rmse_j = 0
        for x in p:
            x_trains, x_tests, y_trains, y_tests = split10items(x, y0, splitss=splitss, random_state=random_state)
            p = 0
            m1 = 0
            for i in range(len(x_trains)):
                RR, RMSE = f_test(x_trains[i], x_tests[i], y_trains[i], y_tests[i])
                p += RR / len(x_trains)
                m1 += RMSE / len(x_trains)
            print("{} RR: {} RMSE: {}".format(k, p, m1), file=mylog)
            if max < p:
                max = p
                max_j = k
            if rmse > m1:
                rmse = m1
                rmse_j = k
            k += 1

        if m < max or abs(max - m) < 0.001:

            b_ = list(bb)
            if m < max:
                m_j = len(bb)
                m = max

        if rm > rmse:
            rm = rmse
            rm_j = len(bb)
        # print("max_RR: {}, delete group is {}".format(max,bb[max_j]), file=mylog)
        print("max_RR: {}, delete group is {}".format(max, bb[rmse_j]), file=mylog)
        # bb.remove(bb[max_j])
        bb.remove(bb[rmse_j])
        print(bb, file=mylog)

    print(file=mylog)
    print('the best groups: {}'.format(b_), file=mylog)
    print("R2_max:{}, b_len: {}".format(m, m_j), file=mylog)
    print("rmse_min: {}, b_len: {}".format(rm, rm_j), file=mylog)
    # print(bb,file=mylog)
    end = time.time()
    print("the spent time is {} seconds".format((end - start)), file=mylog)
    mylog.close()


index = list(range(0, 256))
# index = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 29, 30, 31, 32, 33, 34, 35, 36, 39, 42, 43, 44, 45, 48, 50, 51, 54, 55, 56, 57, 58, 60, 61, 64, 72, 74, 76, 79, 80, 81, 84, 85, 86, 87, 91, 92, 93, 94, 98, 99, 104, 105, 109, 112, 113, 114, 115, 116, 117, 119, 120, 121, 122, 124, 125, 126, 127, 128, 131, 132, 133, 135, 136, 138, 141, 144, 148, 149, 150, 151, 152, 153, 156, 157, 158, 159, 162, 164, 165, 166, 167, 169, 171, 172, 173, 174, 179, 180, 181, 182, 183, 184, 185, 186, 187, 192, 193, 194, 196, 202, 204, 205, 206, 207, 208, 209, 210, 217, 218, 219, 220, 221, 222, 224, 226, 228, 229, 230, 231, 233, 234, 236, 240, 241, 242, 246, 247, 248, 249, 250, 252, 253, 254]
# index = [0, 2, 3, 4, 5, 7, 13, 17, 28, 30, 31, 32, 33, 34, 35, 39, 40, 41, 42, 47, 52, 55, 56, 57, 58, 61, 62, 64, 65, 66, 70, 72, 73, 74, 77, 78, 79, 81, 84, 91, 93, 96, 98, 99, 102, 108, 109, 111, 115, 120, 121, 128, 131, 136, 137, 138, 144, 145, 146, 147, 158, 159, 165, 166, 167, 172, 178, 179, 180, 183, 187, 193, 196, 204, 209, 210, 212, 217, 224, 227, 228, 230, 231, 240]
# index = [0, 3, 8, 13, 29, 30, 32, 33, 34, 35, 39, 40, 42, 47, 52, 55, 57, 60, 61, 64, 66, 69, 72, 74, 77, 79, 91, 92, 93, 95, 102, 107, 108, 109, 115, 120, 121, 128, 136, 137, 138, 145, 146, 158, 159, 166, 167, 170, 172, 178, 179, 180, 183, 187, 191, 193, 204, 209, 210, 212, 224, 227, 230, 231, 240, 241, 246, 252]
# index = [0, 1, 2, 3, 4, 7, 8, 13, 15, 29, 30, 31, 32, 33, 34, 35, 39, 40, 42, 47, 52, 55, 56, 57, 58, 60, 61, 63, 64, 66, 69, 72, 74, 77, 78, 79, 91, 92, 93, 95, 102, 104, 106, 107, 108, 109, 110, 111, 115, 116, 118, 120, 121, 122, 128, 131, 136, 137, 138, 139, 141, 144, 145, 146, 157, 158, 159, 166, 167, 170, 172, 176, 178, 179, 180, 183, 184, 186, 187, 190, 191, 193, 196, 204, 208, 209, 210, 212, 222, 224, 227, 228, 230, 231, 234, 235, 240, 241, 246, 252]
# index = list(range(0, 256))
# index = [1, 4, 5, 6, 10, 12, 14, 21, 22, 30, 32, 40, 42, 48, 52, 53, 58, 59, 60, 67, 72, 73, 81, 84, 85, 87, 88, 93, 94, 98, 100, 101, 106, 108, 109, 110, 116, 117, 119, 120, 121, 122, 125, 128, 134, 148, 156, 157, 158, 159, 164, 166, 170, 194, 196, 199, 204, 205, 207, 210, 211, 219, 221, 223, 224, 227, 231, 233, 234, 235, 236, 240, 241, 250, 251, 252, 254]
# index = [3, 13, 29, 30, 32, 33, 34, 40, 42, 47, 52, 55, 61, 66, 72, 74, 77, 79, 91, 92, 108, 109, 115, 121, 128, 137, 159, 166, 167, 172, 179, 183, 193, 204, 209, 210, 212, 227, 230, 240, 252]
# index = [3, 7, 13, 30, 31, 33, 34, 42, 47, 52, 55, 57, 61, 62, 64, 65, 66, 70, 73, 74, 77, 79, 108, 109, 115, 120, 128, 137, 159, 165, 167, 178, 179, 193, 196, 204, 210, 212, 217, 227, 230, 240]
pre_RPD = 0
def cross(f_test, random_state=11, splitss=10,n_componet=11):


    global index,pre_RPD

    # index =[1, 3, 5, 6, 7, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 32, 34, 35, 42, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 63, 72, 74, 76, 77, 79, 81, 84, 86, 90, 91, 92, 93, 97, 98, 99, 100, 101, 102, 106, 108, 109, 110, 111, 112, 114, 115, 116, 118, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129, 131, 132, 133, 134, 135, 136, 138, 139, 141, 142, 144, 145, 146, 148, 149, 150, 151, 152, 154, 155, 156, 157, 159, 160, 161, 165, 166, 167, 168, 169, 170, 171, 173, 174, 175, 177, 178, 179, 180, 182, 183, 187, 188, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 204, 206, 207, 208, 209, 216, 217, 218, 221, 222, 223, 224, 225, 226, 228, 229, 231, 232, 233, 234, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 250, 251, 252, 253, 254]

    # index =  [3, 7, 13, 30, 31, 33, 34, 42, 47, 52, 55, 57, 61, 62, 64, 65, 66, 70, 73, 74, 77, 79, 108, 109, 115, 120, 128, 137, 159, 165, 167, 178, 179, 193, 196, 204, 210, 212, 217, 227, 230, 240]
    # index =[3, 7, 13, 17, 18, 22, 23, 24, 25, 30, 31, 32, 34, 40, 42, 43, 47, 48, 49, 55, 56, 57, 58, 60, 61, 63, 64, 72, 73, 74, 76, 79, 86, 91, 93, 94, 98, 99, 102, 104, 107, 109, 111, 115, 117, 119, 122, 126, 131, 136, 138, 148, 150, 151, 157, 158, 159, 164, 165, 166, 178, 179, 180, 183, 187, 192, 193, 202, 204, 210, 213, 214, 217, 221, 222, 224, 226, 230, 231, 234, 240, 241, 242, 252, 253]
    # index =[3, 7, 13, 17, 18, 22, 23, 24, 25, 30, 31, 32, 34, 40, 42, 43, 47, 48, 49, 55, 56, 57, 58, 60, 61, 63, 64, 72, 73, 74, 76, 79, 86, 91, 93, 94, 98, 99, 102, 104, 107, 109, 111, 115, 117, 119, 122, 126, 131, 136, 138, 148, 150, 151, 157, 158, 159, 164, 165, 166, 178, 179, 180, 183, 187, 192, 193, 202, 204, 210, 213, 214, 217, 221, 222, 224, 226, 230, 231, 234, 240, 241, 242, 252, 253]



    x0, y0 = loadDataSet01('./PLS-master/data/train.txt', ', ')  # 单因变量与多因变量
    x0 = filter(x0)
    # index = list(range(0, 254))
    index = [2, 3, 6, 12, 20, 23, 24, 29, 33, 41, 46, 47, 50, 51, 54, 55, 56, 57, 60, 61, 63, 65, 71, 73, 74, 76, 78, 88, 90, 92, 103, 106, 108, 114, 118, 119, 120, 121, 122, 123, 125, 127, 130, 155, 156, 157, 158, 165, 171, 172, 174, 175, 176, 177, 178, 179, 195, 196, 199, 201, 202, 203, 212, 216, 223, 224, 225, 229, 231, 237]
    if index is None or "LS_SVM" == f_test.__name__:
        index = list(range(0, x0.shape[1]))

    print(index)
    mm = getDataIndex(x0, index)
    x_trains, x_tests, y_trains, y_tests = split10items(mm, y0, splitss=splitss, random_state=random_state)
    # print(p)
    p = 0
    m = 0

    x1, y1 = loadDataSet01("./PLS-master/data/test.txt", ", ")
    x1 = filter(x1)
    mm1 = getDataIndex(x1, index)
    p1 = 0
    m1 = 0

    for i in range(len(x_trains)):
        if f_test.__name__ .startswith('PCA'):
            a, b, a1, b1 = f_test(x_trains[i], x_tests[i], y_trains[i], y_tests[i], mm1, y1,n_components=n_componet)
        else:
            a, b, a1, b1 = f_test(x_trains[i], x_tests[i], y_trains[i], y_tests[i], mm1, y1)
        p += a
        m += b
        p1 += a1
        m1 += b1
        print("R2",a, 'RMSECV',b)
        # print("r2",a1, 'RMSEP',b1)
    from utils import RPD_total
    ll = len(x_trains)
    R2 = np.round(p / ll * 100,2 )
    RMSECV = m / ll
    r2 = np.round(p1 /ll * 100, 2)
    RMSEP = m1 / ll
    rpd = (RPD_total - pre_RPD)/ ll


    print(u"R2 {0}%".format(R2))
    print(u"RMSECV: {0}".format(RMSECV))
    print(u"r2 {0}%".format(r2))
    print(u"RMSEP: {0}".format(RMSEP))
    print(u"RPD: {0}\n".format(rpd))
    pre_RPD = RPD_total

    return r2,RMSEP



def circle_PCA(f_test,random_state):
    global index
    r2_max = 0
    r2_index = 0
    for i in range(90,120):
        print(i)
        r2,rmsep = cross(f_test,random_state=random_state,n_componet=i)
        if r2_max<r2:
            r2_max=r2
            r2_index=i
    print(r2_index,r2_max)

random_state=11
if __name__ == '__main__':
    import time


    start = time.time()
    random_state = 3
    # main(s_len=1)  #bipls 筛选
    # testRR(index,"./PLS-master/data/L_100_Sigmoid_32_Sigmoid_11_99.44.pkl")

    # cross(regressionNet, random_state=random_state)  # 30000epochs random = 11 spilits=10  R^2 98.23% RMSE. 1.5524621198000335

    # cross(randomForest,random_state=random_state)
    cross(PLS,random_state=random_state)
    # circle_PCA()
    cross(LS_SVM,random_state=random_state)

    # r2,rmsep = cross(PCA,random_state=random_state,n_componet=93)
    # r2,rmsep = cross(PCA_randomForest,random_state=random_state,n_componet=18)
    # circle_PCA(PCA_randomForest,random_state) # 6 效果最好
    # circle_PCA(PCA_LS_SVM,random_state) # [94,103]  98  效果最好
    end = time.time()
    print("the spent time is {} seconds".format((end - start)))

