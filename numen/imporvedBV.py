# -*- coding: utf-8 -*-
'''


'''
import numpy as np
from numen import bitvector as bv

def worst_case_error():
    t = 25
    lower = 0 - t
    upper = 50 + t
    s = 250
    u = upper - lower

    random_list = np.random.uniform(low=lower, high=upper, size=s)
    random_uniform_list = np.asarray([lower+(i+1) * (upper-lower) / (s+1) for i in range(s)])

    run_time = 500
    err_list, err_list_improved = [], []
    for i in range(run_time):
        a = np.random.uniform(low=0, high=50, size=1)[0]
        b = np.random.uniform(low=0, high=50, size=1)[0]
        # print("a=", a, "b=", b)
        true_distance = np.fabs(a-b)

        bv_a = bv.bv_encode(item=a, random_list=random_list, t=t)
        bv_b = bv.bv_encode(item=b, random_list=random_list, t=t)
        estimate_distance = bv.bv_decode(bv_a, bv_b, u)

        ibv_a = bv.bv_encode(item=a, random_list=random_uniform_list, t=t)
        ibv_b = bv.bv_encode(item=b, random_list=random_uniform_list, t=t)
        estimate_distance_imporved = bv.bv_decode(ibv_a, ibv_b, u)

        # print(true_distance, estimate_distance, estimate_distance_imporved)
        err_list.append(np.fabs(true_distance-estimate_distance))
        err_list_improved.append(np.fabs(true_distance-estimate_distance_imporved))

    for i in range(len(err_list)):
        print(err_list[i], err_list_improved[i])


def worst_error_with_s():

    t = 25
    lower = 0 - t
    upper = 50 + t
    u = upper - lower

    for s in range(50, 2001, 50):
        random_list = np.random.uniform(low=lower, high=upper, size=s)
        # random_uniform_list = np.asarray([lower+(i+1) * (upper-lower) / (s+1) for i in range(s)])

        run_time = 4000
        max_err, max_err_improved = 0, 0
        for i in range(run_time):
            a = np.random.uniform(low=0, high=50, size=1)[0]
            b = np.random.uniform(low=0, high=50, size=1)[0]
            # print("a=", a, "b=", b)
            true_distance = np.fabs(a-b)

            bv_a = bv.bv_encode(item=a, random_list=random_list, t=t)
            bv_b = bv.bv_encode(item=b, random_list=random_list, t=t)
            estimate_distance = bv.bv_decode(bv_a, bv_b, u)

            # ibv_a = bv.bv_encode(item=a, random_list=random_uniform_list, t=t)
            # ibv_b = bv.bv_encode(item=b, random_list=random_uniform_list, t=t)
            # estimate_distance_imporved = bv.bv_decode(ibv_a, ibv_b, u)

            if np.fabs(true_distance-estimate_distance) > max_err:
                max_err = np.fabs(true_distance-estimate_distance)
            # if np.fabs(true_distance-estimate_distance_imporved) > max_err_improved:
            #     max_err_improved = np.fabs(true_distance-estimate_distance_imporved)

        print(s, max_err)




if __name__ == "__main__":
    worst_error_with_s()
    pass