import numpy as np

group_num = 10
seednum = 200
t = 50
upper = 100
lower = 0

def needSplit(arr, depth, target_num):
    if len(arr) == 0:
        return False
    length = len(arr[0][1])
    return depth < length and len(arr) > target_num * 2

def inLeft(data, depth, index):
    return data[1][depth] == '0'

def inRight(data, depth, index):
    return data[1][depth] == '1'

def findSplitIndex(arr, used, depth, target_num):
    index = 0
    value = 0
    for i in range(seednum):
        left_num = 0
        right_num = 0
        if used[i] == 1:
            continue
        for ele in arr:
            if inLeft(ele, depth, i):
                left_num += 1
            if inRight(ele, depth, i):
                right_num += 1
        # if left_num >= target_num and right_num >= target_num:
        #     return i
        if left_num * right_num > value:
            index = i
            value = left_num * right_num
    return index


def buildGroup(arr):
    target_num = len(arr) / group_num
    if not needSplit(arr, 0, target_num):
        return arr
    res = []
    stack = []
    stack.append((arr, 0))
    used = [0 for _ in range(seednum)]
    while stack:
        newone = stack[-1]
        stack.pop()
        leftArr = []
        rightArr = []
        if not needSplit(newone[0], newone[1], target_num):
            res.append(newone[0])
        else:
            index = findSplitIndex(newone[0], used, newone[1], target_num)
            for ele in newone[0]:
                if inLeft(ele, newone[1], index):
                    leftArr.append(ele)
                if inRight(ele, newone[1], index):
                    rightArr.append(ele)
            if leftArr:
                stack.append((leftArr, newone[1] + 1))
            if rightArr:
                stack.append((rightArr, newone[1] + 1))
    return res

def