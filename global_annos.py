## 公共区域函数
import pandas as pd
import numpy as np
from global_ import *

def annos():  # 收集有结节图的名字
    annos = pd.read_excel(new_bbox_annos_path)  # 读取bbox_annos.xls
    annos = np.array(annos)  # 读取为数组
    annos = annos.tolist() # 变成列表便于操作
    a = []
    for k in annos:  # 逐行读取
        if len(k) == 3:  # 由于版本不同，有的有头标，有的没有
            jiejie = 2  # 有头标
        else:
            jiejie = 1  # 没头标
        if k[jiejie] != "[]":  # 结节部分不为空的话
            a.append(k[jiejie-1])  # 添加 有结节图的名字到 a
    return a  # 返回所有 有结节的图名
annos = annos()
def str_to_int(aaa):  # str → list
    if aaa == "[]":
        b = []
    else:
        aaa = aaa.lstrip("'[[")
        aaa = aaa.rstrip("]]'")
        b = aaa.split("], [")
        for i in range(len(b)):
            b[i] = b[i].split(",")
            b[i][0] = int(float(b[i][0]))
            b[i][1] = int(float(b[i][1]))
            b[i][2] = int(float(b[i][2]))
            b[i][3] = int(float(b[i][3]))
    return b

def annos_list():  # 收集有结节图的名字
    annos = pd.read_excel(new_bbox_annos_path)  # 读取bbox_annos.xls
    annos = np.array(annos)  # 读取为数组
    annos = annos.tolist() # 变成列表便于操作
    a = []
    for k in annos:  # 逐行读取
        if len(k) == 3:  # 由于版本不同，有的有头标，有的没有
            jiejie = 2  # 有头标
        else:
            jiejie = 1  # 没头标
        if k[jiejie] != "[]":  # 结节部分不为空的话
            a.append([k[jiejie-1],str_to_int(k[jiejie])])  # 添加 有结节图的名字到 a
    return a  # 返回所有 有结节的图名
print(annos_list())
annos_list=annos_list()
