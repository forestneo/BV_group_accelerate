# -*- coding: utf-8 -*-
# @Time    : 2019-03-16 11:35
# @Author  : ForestNeo
# @Email   : dr.forestneo@gmail.com
# @Software: PyCharm

#
import numpy as np
from numen import bitvector as bv


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



if __name__ == '__main__':
    mytest()
    pass




