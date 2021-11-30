# 修改区域

luna_path = r"D:\datasets\LUNA16"
xml_file_path = r'D:\datasets\LIDC-IDRI\LIDC-XML-only\tcia-lidc-xml'
annos_csv = r'D:\datasets\LUNA16\CSVFILES\annotations.csv'
new_bbox_annos_path = r"D:\datasets\sk_output\bbox_annos\bbox_annos.xls"
mask_path = r'D:\datasets\LUNA16\seg-lungs-LUNA16'
output_path = r"D:\datasets\sk_output"
wrong_img_path = r"D:\datasets\wrong_img.xls"

shouci = False
xitong = "linux"  # "windows"


if xitong == "linux":
    fengefu = r"/"
else:
    fengefu = r"\\"


## 公共区域函数
import pandas as pd
import numpy as np

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


