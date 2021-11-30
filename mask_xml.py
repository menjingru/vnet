from global_ import *
from mask_xml_def import for_one_,read_data,resample
import numpy as np
import pandas
import os




anno_name_list = annos()  # 有结节图的名字
print(len(anno_name_list))
wrony = []  # 染色失败的标签

for name in anno_name_list:  # 遍历有结节图的名字
    mask,ct_image_path,wrony = for_one_(name,wrony)  # 输入：单图，空列表wrong，  输出：单图染色mask，mhd文件的绝对地址，错图名字列表wrong
    path = ct_image_path.split("LUNA16")[1].split(".m")[0]  # 取LUNA16后，.mhd前的字符串
    if xitong == "linux":
        path = path.replace(r"\s","/s")
        path = path.replace(r"\1","/1")
    else:
        pass
    print(path)
    # 如 D:\datasets\LUNA16\subset1\1.3.6.1.4.1.14519.5.2.1.6279.6001.287966244644280690737019247886.mhd
    # 则 path =            \subset1\1.3.6.1.4.1.14519.5.2.1.6279.6001.287966244644280690737019247886      一会要用
    ct_image, origin, spacing, isflip = read_data(ct_image_path)  # 读取mhd文件得到图
    ct_image1, origin1, spacing1, isflip1 = read_data(mask_path + fengefu + name+".mhd")  # 读取肺部掩膜
    ct_image1[ct_image1>1]=1  # LUNA16提供的肺部掩膜分左右肺，左肺为3右肺为4，我们需要统一为1
    ct_image = ct_image * ct_image1  # 图与肺mask相乘，肺外区域归0
    image = resample(ct_image, spacing)  # 图 重采样
    msk = resample(mask, spacing)  # 标签 重采样，标签就弄好了。
    print(image.shape)  # 这俩一样大
    print(msk.shape)  # 这俩一样大
       # LUNA16竞赛中常用来做归一化处理的阈值集是-1000和400
    max_num = 400  # 阈值最高
    min_num = -1000  # 阈值最低
    image = (image - min_num) / (max_num - min_num)  # 归一化公式
    image[image > 1] = 1.  # 高于1的归1，float格式
    image[image < 0] = 0.  # 低于0的归0，float格式
    ##   LUNA16竞赛中的均值大约是0.25
    img = image - 0.25  # 去均值，图也弄好了

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_path+fengefu+"bbox_image"):
        os.mkdir(output_path+fengefu+"bbox_image")
    if not os.path.exists(output_path+fengefu+"bbox_mask"):
        os.mkdir(output_path+fengefu+"bbox_mask")
    sub_path = ct_image_path.split("LUNA16")[1].split("1.")[0]  # 取LUNA16后，.mhd前的字符串
    # 如 D:\datasets\LUNA16\subset1\1.3.6.1.4.1.14519.5.2.1.6279.6001.287966244644280690737019247886.mhd
    # 则 path =            \subset1\
    if not os.path.exists(output_path+fengefu+"bbox_image"+sub_path):
        os.mkdir(output_path+fengefu+"bbox_image"+sub_path)
    if not os.path.exists(output_path+fengefu+"bbox_mask"+sub_path):
        os.mkdir(output_path+fengefu+"bbox_mask"+sub_path)


    np.save(output_path+fengefu+"bbox_image"+path,img)  # 图存到 如 D:\datasets\sk_output\bbox_image\subset1\1.3.6.1.4.1.14519.5.2.1.6279.6001.287966244644280690737019247886.npy
    np.save(output_path+fengefu+"bbox_mask"+path,msk)  # 标签存到 如 D:\datasets\sk_output\bbox_mask\subset1\1.3.6.1.4.1.14519.5.2.1.6279.6001.287966244644280690737019247886.npy

wrong_img = pandas.DataFrame(wrony)  # 保存 未染色图
wrong_img.to_excel(wrong_img_path)  # 保存到 wrong_img_path
print("wrony",wrony)  # 打印看一下是否为空