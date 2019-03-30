#!/usr/bin/python
# -*- coding: UTF-8 -*-

# User: ForestNeo
# Time: 2019/3/3 22:04

import numen.bitvector as bv
import numpy as np
import time
from data import reader

def needSplit(data, arr, depth, target_num):
    if len(arr) == 0:
        return False
    length = len(data[0])
    return depth < length and len(arr) > target_num * 1.8


def inLeft(data, count, depth, index):
    return data[count][index] == 0


def inRight(data, count, depth, index):
    return data[count][index] == 1


def findSplitIndex(data, arr, used, depth, target_num):
    return depth
    # index = 0
    # value = 0
    # for i in range(len(data[0])):
    #     left_num = 0
    #     right_num = 0
    #     if used[i] == 1:
    #         continue
    #     for ele in arr:
    #         if inLeft(data, ele, depth, i):
    #             left_num += 1
    #         if inRight(data, ele, depth, i):
    #             right_num +=      # if left_num >= target_num and right_num >= t1
    #     #arget_num:
    #     #     return i
    #     if left_num * right_num > value:
    #         index = i
    #         value = left_num * right_num
    # return index
'''
    在本地就group用于加速，
    @value_list,
    @group_number, 分成几组
    @duplicate，分组的时候中间数据是否需要复制放到两边
    @ret: 返回分组好的列表
'''


def source_group(value_list, group_number, duplicate=True):
    start_time = time.process_time()
    target_num = len(value_list) / group_number
    source = [i for i in range(len(value_list))]
    if not needSplit(value_list, source, 0, target_num):
        return source, 0
    res = []
    stack = [(source, 0)]
    used = [0 for _ in range(len(value_list[0]))]
    while stack:
        newone = stack[-1]
        stack.pop()
        leftArr = []
        rightArr = []
        if not needSplit(value_list, newone[0], newone[1], target_num):
            if newone[0]:
                res.append(newone[0])
        else:
            index = findSplitIndex(value_list, newone[0], used, newone[1], target_num)
            for ele in newone[0]:
                if duplicate:
                    if inLeft(value_list, ele, newone[1], index):
                        leftArr.append(ele)
                    if inRight(value_list, ele, newone[1], index):
                        rightArr.append(ele)
                else:
                    if inLeft(value_list, ele, newone[1], index):
                        leftArr.append(ele)
                    elif inRight(value_list, ele, newone[1], index):
                        rightArr.append(ele)
            if leftArr:
                stack.append((leftArr, newone[1] + 1))
            if rightArr:
                stack.append((rightArr, newone[1] + 1))
    end_time = time.process_time()
    used_time = end_time - start_time
    return res, used_time


def anonymize_group(value_bv_list: np.ndarray, duplicatie=True):
    if not len(value_bv_list.shape) == 2:
        exit(0)
    flag_num = np.sum(value_bv_list, axis=0)

    ret = set()
    return ret


def compare_with_group(value_bv_list_a, value_bv_list_b, threshold, u, threshold_times, times, group_number):
    start_time = time.process_time()
    ret_true = set()
    ret_false = set()
    value_bv_group_a, gt1 = source_group(value_bv_list_a, group_number)
    value_bv_group_b, gt2 = source_group(value_bv_list_b, group_number)
    for ga in value_bv_group_a:
        for gb in value_bv_group_b:
            dist = bv.bv_decode(value_bv_list_a[ga[0]], value_bv_list_b[gb[0]], u)
            if dist <= times * threshold * threshold_times:
                for m in ga:
                    for n in gb:
                        dist = bv.bv_decode(value_bv_list_a[m], value_bv_list_b[n], u)
                        if dist <= threshold * threshold_times:
                            ret_true.add((m, n))
                        else:
                            ret_false.add((m, n))
    end_time = time.process_time()
    used_time = end_time - start_time
    set_list = anonymize_group(value_bv_list_a)
    return ret_true, ret_false, used_time, gt1 + gt2


def compare_without_group(value_bv_list_a, value_bv_list_b, threshold, u, threshold_times=1.2):
    # 不采用分组策略，直接用BV编码的结果进行匹配
    start_time = time.process_time()
    ret_true = set()
    ret_false = set()
    for i in range(len(value_bv_list_a)):
        for j in range(len(value_bv_list_b)):
            dist = bv.bv_decode(value_bv_list_a[i], value_bv_list_b[j], u)
            if dist <= threshold * threshold_times:
                ret_true.add((i, j))
            else:
                ret_false.add((i, j))
    end_time = time.process_time()
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
    if not tp_set:
        return 0, 0, 0
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
    age_list = reader.read_age("../data/test.txt")
    print(max(age_list))
    print(min(age_list))
    print(len(age_list))

    t = 50
    lower = 0 - t
    upper = 100 + t
    s = 200
    u = upper - lower

    list_number = 3000
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
        ret_true, ret_false, time_without_group_2 = compare_without_group(value_bv_list_a, value_bv_list_b, threshold, u, threshold_times)
        precision_ibv, recall_ibv, fscore_ibv = analyse_result(groundtruth_true, groundtruth_false, ret_true, ret_false)

        print(threshold_times, precision_bv, recall_bv, fscore_bv, precision_ibv, recall_ibv, fscore_ibv, time_without_group)

def record_linkage_with_age_dataset_group():
    age_list = reader.read_age("../data/test.txt")
    print(max(age_list))
    print(min(age_list))
    print(len(age_list))

    t = 50
    lower = 0 - t
    upper = 100 + t
    s = 200
    u = upper - lower

    list_number = 3000
    value_list_a = age_list[0:list_number]
    value_list_b = age_list[list_number+1:2*list_number]

    threshold = 5
    threshold_times_list = [0.8 + 0.05*i for i in range(11)]
    for threshold_times in threshold_times_list:
        random_list = np.random.uniform(low=lower, high=upper, size=s)
        value_bv_list_a = np.asarray([bv.bv_encode(item=value_list_a[i], random_list=random_list, t=t) for i in range(len(value_list_a))])
        value_bv_list_b = np.asarray([bv.bv_encode(item=value_list_b[i], random_list=random_list, t=t) for i in range(len(value_list_b))])
        groundtruth_true, groundtruth_false = get_groundtruth(value_list_a, value_list_b, threshold)
        ret_true, ret_false, time_without_group = compare_with_group(value_bv_list_a, value_bv_list_b, threshold, u, threshold_times)
        precision_bv, recall_bv, fscore_bv = analyse_result(groundtruth_true, groundtruth_false, ret_true, ret_false)

        random_uniform_list = np.asarray([lower + (i + 1) * (upper - lower) / (s + 1) for i in range(s)])
        value_bv_list_a = np.asarray([bv.bv_encode(item=value_list_a[i], random_list=random_uniform_list, t=t) for i in range(len(value_list_a))])
        value_bv_list_b = np.asarray([bv.bv_encode(item=value_list_b[i], random_list=random_uniform_list, t=t) for i in range(len(value_list_b))])
        groundtruth_true, groundtruth_false = get_groundtruth(value_list_a, value_list_b, threshold)
        ret_true, ret_false, time_without_group = compare_without_group(value_bv_list_a, value_bv_list_b, threshold, u, threshold_times)
        precision_ibv, recall_ibv, fscore_ibv = analyse_result(groundtruth_true, groundtruth_false, ret_true, ret_false)

        print(threshold_times, precision_bv, recall_bv, fscore_bv, precision_ibv, recall_ibv, fscore_ibv, time_without_group)

def record_linkage_with_age_dataset_threshold():
    age_list = reader.read_age("../data/test.txt")
    t = 50
    lower = 0 - t
    upper = 100 + t
    s = 200
    u = upper - lower

    list_number = 1000
    value_list_a = age_list[0:list_number]
    value_list_b = age_list[list_number+1:2*list_number]

    threshold_list = [5]
    group_number_list = [i for i in range(2, 11, 2)]
    times_list = [i / 10 for i in range(2, 32, 2)]
    threshold_times = 1.1
    for threshold in threshold_list:
        random_list = np.random.uniform(low=lower, high=upper, size=s)
        value_bv_list_a = np.asarray([bv.bv_encode(item=value_list_a[i], random_list=random_list, t=t) for i in range(len(value_list_a))])
        value_bv_list_b = np.asarray([bv.bv_encode(item=value_list_b[i], random_list=random_list, t=t) for i in range(len(value_list_b))])
        groundtruth_true, groundtruth_false = get_groundtruth(value_list_a, value_list_b, threshold)
        ret_true, ret_false, time_without_group = compare_without_group(value_bv_list_a, value_bv_list_b, threshold, u, threshold_times)
        precision_bv, recall_bv, fscore_bv = analyse_result(groundtruth_true, groundtruth_false, ret_true, ret_false)
        print(precision_bv, recall_bv, fscore_bv, time_without_group)
        for group_number in group_number_list:
            for times in times_list:
                print(group_number, times)
                groundtruth_true, groundtruth_false = get_groundtruth(value_list_a, value_list_b, threshold)
                ret_true, ret_false, time_with_group, group_time = compare_with_group(value_bv_list_a, value_bv_list_b, threshold, u, threshold_times, times, group_number)
                precision_bv, recall_bv, fscore_bv = analyse_result(groundtruth_true, groundtruth_false, ret_true, ret_false)
                print(precision_bv, recall_bv, fscore_bv, time_with_group, group_time)

        # random_uniform_list = np.asarray([lower + (i + 1) * (upper - lower) / (s + 1) for i in range(s)])
        # value_bv_list_a = np.asarray([bv.bv_encode(item=value_list_a[i], random_list=random_uniform_list, t=t) for i in range(len(value_list_a))])
        # value_bv_list_b = np.asarray([bv.bv_encode(item=value_list_b[i], random_list=random_uniform_list, t=t) for i in range(len(value_list_b))])
        # groundtruth_true, groundtruth_false = get_groundtruth(value_list_a, value_list_b, threshold)
        # ret_true, ret_false, time_without_group = compare_without_group(value_bv_list_a, value_bv_list_b, threshold, u, threshold_times)
        # precision_ibv, recall_ibv, fscore_ibv = analyse_result(groundtruth_true, groundtruth_false, ret_true, ret_false)
        # groundtruth_true, groundtruth_false = get_groundtruth(value_list_a, value_list_b, threshold)
        # print(precision_ibv, recall_ibv, fscore_ibv, time_without_group)
        # ret_true, ret_false, time_with_group, group_time = compare_with_group(value_bv_list_a, value_bv_list_b, threshold, u, threshold_times)
        # precision_ibv, recall_ibv, fscore_ibv = analyse_result(groundtruth_true, groundtruth_false, ret_true, ret_false)
        # print(precision_ibv, recall_ibv, fscore_ibv, time_with_group)

def record_linkage_with_age_dataset_threshold_group():
    age_list = reader.read_age("../data/test.txt")
    print(max(age_list))
    print(min(age_list))
    print(len(age_list))

    t = 50
    lower = 0 - t
    upper = 100 + t
    s = 200
    u = upper - lower

    list_number = 1000
    value_list_a = age_list[0:list_number]
    value_list_b = age_list[list_number+1:2*list_number]

    threshold_list = [3,4,5,6,7,8,9,10]
    threshold_times = 1.1
    for threshold in threshold_list:
        random_list = np.random.uniform(low=lower, high=upper, size=s)
        value_bv_list_a = np.asarray([bv.bv_encode(item=value_list_a[i], random_list=random_list, t=t) for i in range(len(value_list_a))])
        value_bv_list_b = np.asarray([bv.bv_encode(item=value_list_b[i], random_list=random_list, t=t) for i in range(len(value_list_b))])
        groundtruth_true, groundtruth_false = get_groundtruth(value_list_a, value_list_b, threshold)
        ret_true, ret_false, time_with_group1 = compare_with_group(value_bv_list_a, value_bv_list_b, threshold, u, threshold_times)
        precision_bv, recall_bv, fscore_bv = analyse_result(groundtruth_true, groundtruth_false, ret_true, ret_false)

        random_uniform_list = np.asarray([lower + (i + 1) * (upper - lower) / (s + 1) for i in range(s)])
        value_bv_list_a = np.asarray([bv.bv_encode(item=value_list_a[i], random_list=random_uniform_list, t=t) for i in range(len(value_list_a))])
        value_bv_list_b = np.asarray([bv.bv_encode(item=value_list_b[i], random_list=random_uniform_list, t=t) for i in range(len(value_list_b))])
        groundtruth_true, groundtruth_false = get_groundtruth(value_list_a, value_list_b, threshold)
        ret_true, ret_false, time_with_group2 = compare_with_group(value_bv_list_a, value_bv_list_b, threshold, u, threshold_times)
        precision_ibv, recall_ibv, fscore_ibv = analyse_result(groundtruth_true, groundtruth_false, ret_true, ret_false)

        print(threshold, precision_bv, recall_bv, fscore_bv, precision_ibv, recall_ibv, fscore_ibv, time_with_group1, time_with_group2)

if __name__ == "__main__":
    record_linkage_with_age_dataset_threshold()
    #record_linkage_with_age_dataset_threshold_group()
    pass



