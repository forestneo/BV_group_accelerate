# -*- coding: utf-8 -*-
'''


'''

import numpy as np

def read_medical_age(filename):
    f = open(filename, 'r')
    value_list = f.readlines()
    f.close()

    ret = []
    for item in value_list[0:-1]:
        # print(item, type(item))
        if len(item.split('.')) != 2:
            continue
        if item.split('\n')[0] == ' ':
            continue
        value = int(float(item.split("\n")[0]))
        if value < 100:
            ret.append(value)
    return np.asarray(ret)


if __name__ == "__main__":
    print(read_medical_age())
    pass