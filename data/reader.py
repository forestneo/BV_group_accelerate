# -*- coding: utf-8 -*-
'''


'''

import numpy as np

def read_age(filename):
    f = open(filename, "r")
    data = f.read().split()
    f.close()
    res = []
    for d in data:
        try:
            res.append(int(float(d)))
        except:
            pass
    return res

if __name__ == "__main__":
    print(read_age("test.txt"))
    pass