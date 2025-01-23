## 公共区域函数
import pandas as pd
import numpy as np
from global_ import new_bbox_annos_path


def annos():  # 收集有结节图的名字
    annos = pd.read_excel(new_bbox_annos_path)  # 读取bbox_annos.xls
    a = []
    for ind,val in annos.iterrows():
        if val['annos'] != '[]':
            a.append(val['name'])
    return a  # 返回所有 有结节的图名
annos = annos()
