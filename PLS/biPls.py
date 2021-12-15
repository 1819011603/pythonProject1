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

def main(f_test=PLS,s_len = 11,splitss = 10,random_state=11):
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
            x_trains, x_tests, y_trains, y_tests = split10items(x, y0, splitss=splitss, random_state=random_state)
            p = 0
            m1 = 0
            for i in range(len(x_trains)):
                RR, RMSE = f_test(x_trains[i], x_tests[i], y_trains[i], y_tests[i])
                p += RR/len(x_trains)
                m1 += RMSE/len(x_trains)
            print("{} RR: {} RMSE: {}".format(k, p, m1), file=mylog)
            if max < p:
                max = p
                max_j = k
            if rm > m1:
                rm = m1
                rm_j = len(bb)
            k += 1

        if m < max or abs(max-m) < 0.001:


            b_ = list(bb)
            if m < max:
                m_j = len(bb)
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



def cross(f_test,random_state=11,splitss=10):
    # index =[0, 2, 3, 4, 5, 7, 8, 13, 17, 28, 30, 31, 32, 33, 34, 35, 39, 40, 41, 42, 47, 52, 55, 56, 57, 58, 61, 62, 64, 65, 66, 70, 72, 73, 74, 77, 78, 79, 81, 84, 91, 93, 96, 98, 99, 102, 104, 106, 108, 109, 110, 111, 115, 118, 120, 121, 128, 131, 136, 137, 138, 141, 144, 145, 146, 147, 158, 159, 165, 166, 167, 172, 176, 178, 179, 180, 183, 187, 191, 193, 196, 204, 209, 210, 212, 217, 224, 227, 228, 230, 231, 234, 240]
    index = [0, 2, 3, 4, 5, 7, 13, 17, 28, 30, 31, 32, 33, 34, 35, 39, 40, 41, 42, 47, 52, 55, 56, 57, 58, 61, 62, 64, 65, 66, 70, 72, 73, 74, 77, 78, 79, 81, 84, 91, 93, 96, 98, 99, 102, 108, 109, 111, 115, 120, 121, 128, 131, 136, 137, 138, 144, 145, 146, 147, 158, 159, 165, 166, 167, 172, 178, 179, 180, 183, 187, 193, 196, 204, 209, 210, 212, 217, 224, 227, 228, 230, 231, 240]
    x0, y0 = loadDataSet01('./PLS-master/data/test_all_reflect1.txt', ', ')  # 单因变量与多因变量
    x0 = filter(x0)
    l1 = np.shape(x0)[0]
    mm = []
    for i in range(l1):
        mm.append(np.array(x0[i][index]))
    mm = np.array(mm)
    x_trains, x_tests, y_trains, y_tests = split10items(mm, y0, splitss=splitss, random_state=random_state)
        # print(p)
    p = 0
    m = 0
    for i in range(len(x_trains)):
        a, b = f_test(x_trains[i], x_tests[i], y_trains[i], y_tests[i])
        p += a
        m += b

    print(u"R^2 {0}%".format(np.round(p / len(x_trains) * 100, 2)))
    print(u"RMSE.", m / len(x_trains))

if __name__ == '__main__':
    import time
    start = time.time()
    random_state=65
    # main(s_len=1)
    cross(regressionNet,random_state=random_state)  # 30000epochs random = 11 spilits=10  R^2 98.23% RMSE. 1.5524621198000335
    cross(PLS,random_state=random_state)
    end = time.time()
    print("the spent time is {} seconds".format((end - start)))