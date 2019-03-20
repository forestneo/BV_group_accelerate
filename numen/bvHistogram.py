# -*- coding: utf-8 -*-
# @Time    : 2019-03-11 15:27
# @Author  : ForestNeo
# @Email   : dr.forestneo@gmail.com
# @Software: PyCharm

#

import numpy as np
import random
from numen import bitvector as bv
from data import reader
import matplotlib.pyplot as plt

def restore_data(bv_list, u):
    l = len(bv_list)
    origin = np.zeros((l, ), dtype=np.int)
    found = False
    zero_index = 0
    for i in range(l):
        for j in range(i + 1, l):
            if bv.bv_decode(bv_list[i], bv_list[j], u) > 97:
                origin[i] = 0
                origin[j] = 100
                found = True
                zero_index = i
                break
        if found:
            break
    for i in range(l):
        origin[i] = bv.bv_decode(bv_list[zero_index], bv_list[i], u)
    return origin

if __name__ == '__main__':
    datalist = np.asarray(reader.read_age("../data/test.txt"))
    datalist = datalist[np.where(datalist < 100)]
    # print(datalist)
    print(len(datalist))
    # plt.hist(datalist)
    # plt.show()
    t = 50
    lower = 0 - t
    upper = 100 + t
    s = 2000
    u = upper - lower
    random_list = np.random.uniform(low=lower, high=upper, size=s)
    bv_list = [bv.bv_encode(item=datalist[i], random_list=random_list, t=t) for i in range(len(datalist))]
    l = len(bv_list)
    print(l)
    origin = restore_data(bv_list, u)
    print(origin)
    plt.hist(origin)
    plt.show()
    pass




