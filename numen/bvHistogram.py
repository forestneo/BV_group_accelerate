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

if __name__ == '__main__':
    datalist = np.asarray(reader.read_age("../data/test.txt"))
    datalist = datalist[np.where(datalist <= 100)]

    t = 50
    lower = 0 - t
    upper = 100 + t
    s = 2000
    u = upper - lower
    random_list = np.random.uniform(low=lower, high=upper, size=s)

    bv_list = [bv.bv_encode(item=datalist[i], random_list=random_list, t=t) for i in range(len(datalist))]

    # dist[i,j] = bv.bv_decode(bv_list[i], bv_list[j])

    pass




