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


def onehot(v, u=100):
    if not (v >= 0 or v < u):
        print("in onehot encoding, error")
        exit(0)
    y = np.zeros([u+1], dtype=int)
    y[v] = 1
    return y


def perturbate(p, arr):
    ret = np.zeros([len(p)], dtype=int)
    for index in p:
        ret[index] = arr[p[index]]
    return ret

def bvExample_experiment():
    t = 25
    lower = 0 - t
    upper = 50 + t
    s = 512
    u = upper - lower

    value_list = [random.randint(0, 25) for i in range(2)]
    random_list = np.random.uniform(low=lower, high=upper, size=s)

    x, y = 34, 40
    bv_x = bv_encode(item=x, random_list=random_list, t=t)
    bv_y = bv_encode(item=y, random_list=random_list, t=t)
    print("x, y = 34, 40")
    print("bv_x = " + str(bv_x))
    print("bv_y = " + str(bv_y))
    true_distance = np.fabs(x-y)
    estimate_distance_bv = bv_decode(bv_x, bv_y, u)

    P = np.random.permutation(s)

    random_uniform_list = np.asarray([lower+(i+1) * (upper-lower) / (s+1) for i in range(s)])
    ibv_x = bv_encode(item=x, random_list=random_uniform_list, t=t)
    ibv_y = bv_encode(item=y, random_list=random_uniform_list, t=t)
    estimate_distance_ibv = bv_decode(ibv_x, ibv_y, u)
    print("ibv_x" + str(perturbate(P, ibv_x)))
    print("ibv_y" + str(perturbate(P, ibv_y)))
    print(true_distance, estimate_distance_bv, estimate_distance_ibv)


def experiment():
    t = 25
    lower = 0 - t
    upper = 50 + t
    s = 120
    u = upper - lower

    for i in range(1):
        random_uniform_list = np.asarray([lower + (i + 1) * (upper - lower) / (s + 1) for i in range(s)])

        x, y = 34, 40
        bv_x = bv_encode(item=x, random_list=random_uniform_list, t=t)
        bv_y = bv_encode(item=y, random_list=random_uniform_list, t=t)
        estimate_distance_bv = bv_decode(bv_x, bv_y, u)
        print(estimate_distance_bv)

if __name__ == "__main__":
    # 展示编码结果
    # bvExample_experiment()
    experiment()
    pass
