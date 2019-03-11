#!/usr/bin/python
# -*- coding: UTF-8 -*-

# User: ForestNeo
# Time: 2019/3/3 22:04

from numen import bitvector as bv
import numpy as np
import time
from data import reader

'''
    在本地就group用于加速，
    @value_list,
    @group_number, 分成几组
    @duplicate，分组的时候中间数据是否需要复制放到两边
    @ret: 返回set，set每个元素是set，表示那一组的编号。比如十条纪录分了两组{{1,2,3,4,6}{0,5,7,8,9}}
'''
def source_group(value_list, group_number, duplicate=True):
    pass


def anonymize_group(value_bv_list: np.ndarray, duplicatie=True):
    if not len(value_bv_list.shape) == 2:
        exit(0)
    flag_num = np.sum(value_bv_list, axis=0)
    print(flag_num)

    ret = set()
    return ret


def compare_with_group(value_bv_list_a, value_bv_list_b, threshold, threshold_times=1.2):
    set_list = anonymize_group(value_bv_list_a)
    return set_list


def compare_without_group(value_bv_list_a, value_bv_list_b, threshold, u, threshold_times=1.2):
    # 不采用分组策略，直接用BV编码的结果进行匹配
    start_time = time.clock()
    ret_true = set()
    ret_false = set()
    for i in range(len(value_bv_list_a)):
        for j in range(len(value_bv_list_b)):
            dist = bv.bv_decode(value_bv_list_a[i], value_bv_list_b[j], u)
            if dist <= threshold * threshold_times:
                ret_true.add((i, j))
            else:
                ret_false.add((i, j))
    end_time = time.clock()
    used_time = end_time - start_time
    return ret_true, ret_false, used_time


def get_groundtruth(value_list_a, value_list_b, threshold):
    ret_true = set()
    ret_false = set()
    for i in range(len(value_list_a)):
        for j in range(len(value_list_b)):
            if np.fabs(value_list_a[i] - value_list_b[j]) <= threshold:
                ret_true.add((i, j))
            else:
                ret_false.add((i, j))
    return ret_true, ret_false


# 计算准确率，召回率，f-score
def analyse_result(groundtruth_true: set, groundtruth_false: set, compare_truth: set, compare_false: set):
    tp_set = compare_truth.intersection(groundtruth_true)
    tn_set = compare_false.intersection(groundtruth_false)
    fp_set = compare_truth.intersection(groundtruth_false)
    fn_set = compare_false.intersection(groundtruth_true)

    precision = len(tp_set) / (len(tp_set) + len(fp_set))
    recall = len(tp_set) / (len(tp_set) + len(fn_set))
    fscore = 2 * precision * recall / (precision + recall)
    return precision, recall, fscore


def my_test():
    t = 25
    lower = 0 - t
    upper = 50 + t
    s = 2000
    u = upper - lower

    len_list_a = 50
    len_list_b = 100
    value_list_a = np.random.randint(0, 25, len_list_a)
    value_list_b = np.random.randint(0, 25, len_list_b)
    random_list = np.random.uniform(low=lower, high=upper, size=s)

    value_bv_list_a = np.asarray([bv.bv_encode(item=value_list_a[i], random_list=random_list, t=t) for i in range(len(value_list_a))])
    value_bv_list_b = np.asarray([bv.bv_encode(item=value_list_b[i], random_list=random_list, t=t) for i in range(len(value_list_b))])

    threshold = 5
    threshold_times = 1.15
    groundtruth_true, groundtruth_false = get_groundtruth(value_list_a, value_list_b, threshold)
    ret_true, ret_false, time_without_group = compare_without_group(value_bv_list_a, value_bv_list_b, threshold, u, threshold_times)
    print("time caused in compare_without_group : " + str(time_without_group) + " seconds")

    precision, recall, fscore = analyse_result(groundtruth_true, groundtruth_false, ret_true, ret_false)
    print("precision = " + str(precision))
    print("recall = " + str(recall))
    print("fscore = " + str(fscore))

    compare_with_group(value_bv_list_a, value_bv_list_b, threshold, threshold_times)



def record_linkage_with_age_dataset():
    age_list = reader.read_medical_age("../data/medical_data.txt")
    # print(max(age_list))
    # print(min(age_list))
    # print(len(age_list))

    t = 50
    lower = 0 - t
    upper = 100 + t
    s = 200
    u = upper - lower

    list_number = 300
    value_list_a = age_list[0:list_number]
    value_list_b = age_list[list_number+1:2*list_number]

    threshold = 5
    threshold_times_list = [0.8 + 0.05*i for i in range(11)]
    for threshold_times in threshold_times_list:

        random_list = np.random.uniform(low=lower, high=upper, size=s)
        value_bv_list_a = np.asarray([bv.bv_encode(item=value_list_a[i], random_list=random_list, t=t) for i in range(len(value_list_a))])
        value_bv_list_b = np.asarray([bv.bv_encode(item=value_list_b[i], random_list=random_list, t=t) for i in range(len(value_list_b))])
        groundtruth_true, groundtruth_false = get_groundtruth(value_list_a, value_list_b, threshold)
        ret_true, ret_false, time_without_group = compare_without_group(value_bv_list_a, value_bv_list_b, threshold, u, threshold_times)
        precision_bv, recall_bv, fscore_bv = analyse_result(groundtruth_true, groundtruth_false, ret_true, ret_false)

        random_uniform_list = np.asarray([lower + (i + 1) * (upper - lower) / (s + 1) for i in range(s)])
        value_bv_list_a = np.asarray([bv.bv_encode(item=value_list_a[i], random_list=random_uniform_list, t=t) for i in range(len(value_list_a))])
        value_bv_list_b = np.asarray([bv.bv_encode(item=value_list_b[i], random_list=random_uniform_list, t=t) for i in range(len(value_list_b))])
        groundtruth_true, groundtruth_false = get_groundtruth(value_list_a, value_list_b, threshold)
        ret_true, ret_false, time_without_group = compare_without_group(value_bv_list_a, value_bv_list_b, threshold, u, threshold_times)
        precision_ibv, recall_ibv, fscore_ibv = analyse_result(groundtruth_true, groundtruth_false, ret_true, ret_false)

        print(threshold_times, precision_bv, recall_bv, fscore_bv, precision_ibv, recall_ibv, fscore_ibv)


if __name__ == "__main__":
    record_linkage_with_age_dataset()
    pass



