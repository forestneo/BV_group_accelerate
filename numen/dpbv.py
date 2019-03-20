# -*- coding: utf-8 -*-
# @Time    : 2019-03-16 11:35
# @Author  : ForestNeo
# @Email   : dr.forestneo@gmail.com
# @Software: PyCharm

#
import numpy as np
from numen import bitvector as bv
from data import reader

def mytest():
    t = 25
    lower = 0 - t
    upper = 50 + t
    u = upper - lower
    s = 10000
    random_list = np.random.uniform(low=lower, high=upper, size=s)


    times = 100
    for i in range(times):
        a = np.random.uniform(low=0, high=50, size=1)[0]
        b = np.random.uniform(low=0, high=50, size=1)[0]
        true_distance = np.fabs(a-b)

        bv_a = bv.bv_encode(item=a, random_list=random_list, t=t)
        bv_b = bv.bv_encode(item=b, random_list=random_list, t=t)
        estimate_distance = bv.bv_decode(bv_a, bv_b, u)

        print("true distance = " + str(true_distance))
        print("estimate distance = " + str(estimate_distance))


def get_original_hist(datalist):
    res = np.zeros([101], dtype=int)
    for data in datalist:
        res[data] = res[data] + 1
    return res


def dpbv_for_hist(datalist, p):
    N = len(datalist)
    onehot_list = np.asarray([bv.onehot(datalist[i]) for i in range(N)])
    dpbv_list = np.asarray([bv.coinflip(vec=onehot_list[i], p=p) for i in range(len(onehot_list))])

    M1 = np.sum(dpbv_list, axis=0)
    M = (M1 - N*(1-p))/(2*p-1)
    return M


def error_of_hist(hist_a, hist_b):
    if len(hist_a) != len(hist_b):
        print("the length of hist is not the same")
        exit(0)
    err1 = max([np.fabs(hist_a[i] - hist_b[i]) for i in range(len(hist_a))])
    err2 = max([np.fabs(hist_a[i] - hist_b[len(hist_a)-i-1]) for i in range(len(hist_a))])
    return min(err1, err2)


def ibv_hist_test():
    t = 50
    lower = 0 - t
    upper = 100 + t
    s = 1000
    u = upper-lower


    # datalist = np.asarray(reader.read_age("../data/test.txt"))
    # datalist = datalist[np.where(datalist <= 100)]
    # datalist = datalist[:500]
    datalist = [i for i in range(100)]
    print(str(len(datalist)) + " records have been loaded! With max = " + str(max(datalist)) + ", min = " + str(min(datalist)) + "!")
    print(datalist)

    random_list = np.random.uniform(low=lower, high=upper, size=s)
    # random_list = np.asarray([lower+(i+1) * (upper-lower) / (s+1) for i in range(s)])

    ibv_list = [bv.bv_encode(datalist[i], random_list, t) for i in range(len(datalist))]
    dist_matrix = np.zeros(shape=[len(datalist), len(datalist)])

    i_index = 0
    j_index = 0
    max_index = 0
    for i in range(len(datalist)):
        for j in range(i, len(datalist)):
            dist_matrix[i][j] = bv.bv_decode(ibv_list[i], ibv_list[j], u)
            dist_matrix[j][i] = dist_matrix[i][j]
            if dist_matrix[i][j] >= max_index:
                i_index, j_index, max_index = i, j, dist_matrix[i][j]
    print(dist_matrix)
    print(i_index, j_index, max_index)

    # mark i_index as min of data, j_index as max of data
    estimate_data_list = []
    for i in range(len(ibv_list)):
        estimate_value = int(0 + (dist_matrix[i][i_index])/(max_index) * 100)
        estimate_data_list.append(estimate_value)

    original_hist = get_original_hist(datalist)
    estimate_hist = get_original_hist(estimate_data_list)

    print("original_hist:" + str(original_hist))
    print("estimate_hist:" + str(estimate_hist))
    print("error_of_hist:" + str(error_of_hist(original_hist, estimate_hist)))




# 利用估计的直方图来计算均值，数据分布从0-100
def get_mean_from_hist(hist):
    ret = 0
    for i in range(len(hist)):
        ret = ret + i * hist[i]
    return 1.0 * ret / np.sum(hist)


def generate_normal_dist(size):
    data = np.random.normal(loc=50, scale=20, size=size)
    data = data[np.where(data <= 100)]
    data = data[np.where(data >= 0)]
    data = np.asarray([int(b) for b in data])
    return data


def mean_estimation_with_hist():
    datalist = np.asarray(reader.read_age("../data/test.txt"))
    datalist = datalist[np.where(datalist <= 100)]
    print(str(len(datalist)) + " records have been loaded! With max = " + str(max(datalist)) + ", min = " + str(min(datalist)) + "!")
    original_hist = get_original_hist(datalist)
    print("original hist = \n", original_hist)
    estimate_hist = dpbv_for_hist(datalist, p=0.9)
    print("estimate hist = \n", estimate_hist)

    print("original mean = " + str(get_mean_from_hist(original_hist)))
    print("estimate mean = " + str(get_mean_from_hist(estimate_hist)))


def hist_with_normal_distribute():
    size = 10000
    datalist = generate_normal_dist(size)
    original_hist = get_original_hist(datalist)
    print("--------------------- original hist = \n", original_hist)
    estimate_hist = dpbv_for_hist(datalist, p=0.9)
    print("--------------------- estimate hist = \n", estimate_hist)

    for i in range(len(original_hist)):
        print(i, original_hist[i], estimate_hist[i])

    print("original mean = " + str(get_mean_from_hist(original_hist)))
    print("estimate mean = " + str(get_mean_from_hist(estimate_hist)))


if __name__ == '__main__':
    ibv_hist_test()




