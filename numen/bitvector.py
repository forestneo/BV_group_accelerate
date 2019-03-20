# -*- coding: UTF-8 -*-

# User: ForestNeo
# Time: 2019/03/03 9:31

import numpy as np
import random
import pickle
import time


def bv_hashing(item, random_variable, t):
    return random_variable - t <= item <= random_variable + t


def bv_encode(item, random_list, t):
    ret = [bv_hashing(item, random_variable, t) for random_variable in random_list]
    return np.asarray(ret, dtype=int)


def bv_decode(bv_1, bv_2, u):
    if bv_1.size != bv_2.size:
        raise Exception(ValueError)
    if not len(bv_1.shape) == 1:
        raise TypeError("the type of bv is wrong")
    s = bv_1.size
    dh = np.sum(np.fabs(bv_1 - bv_2), dtype=int)  # 计算汉明距离
    de = 1.0 * dh * u / (2 * s)  # 估计欧式距离
    return de


def coinflip(vec, p):
    flip = np.random.binomial(1, p, len(vec))
    return(np.where(flip==1, vec, 1-vec))

def my_test():
    t = 25
    lower = 0 - t
    upper = 50 + t
    s = 2000
    u = upper - lower

    value_list = [random.randint(0,25) for i in range(2)]
    random_list = np.random.uniform(low=lower, high=upper, size=s)

    value_rbv_list = [bv_encode(item=value_list[i], random_list=random_list, t=t) for i in range(len(value_list))]

    print(value_list)
    print(random_list)

    true_distance = np.fabs(value_list[0] - value_list[1])
    estimate_distance = bv_decode(value_rbv_list[0], value_rbv_list[1], u=u)
    print(true_distance, estimate_distance)


def onehot(v, u=100):
    if not (v >= 0 or v < u):
        print("in onehot encoding, error")
        exit(0)
    y = np.zeros([u+1], dtype=int)
    y[v] = 1
    return y


if __name__ == "__main__":
    print(onehot(1, 10))
    pass
