from global_ import *
import xml
from xml.dom.minidom import parse  # 用来读xml文件的
import numpy as np
import cv2 as cv   # 用来做形态填充的
import scipy
from scipy import ndimage   # 用来补洞的
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.ndimage.interpolation import zoom


# 在开始之前，由于我们需要只保留有结节的图像，因此从annotations中取得有结节的图像名字。①进行结节中心的坐标变换，并保存到 bbox_annos
#                                                                        ②提取结节不为空的图的名字 annos()  这个在主函数

def get_raw_label(img_name,img,annos,origin,zheng):  # 进行单图的坐标转换，把结节坐标 从世界坐标 准换到 图像坐标 （[1,1,1]像距） 输入为：图的名字，图，LUNA6中的annotations，像素间隔，是否翻转
    annos_these_list = []    # 准备装[[名字,结节1],[名字,结节2],...] 装所有有结节的图的名字和坐标点（原始）
    for i in range(len(annos)):   # 遍历annotations所有数据
        if annos[i][0] == img_name:   ###  如果名字相符
            annos_these_list.append(list(annos[i]))   ###  装进去
    print(annos_these_list)  # [["名字",x,y,z,diam],["名字",x,y,z,diam]] ,其中，所有的名字都与img_name相符，因为一个图可能有好几个结节坐标。diam是直径
    return_list = []  # 准备装 ["名字", [[结节1],[结节2],...] ] ，装所有有结节的图的名字和坐标点（原始→坐标变换后）
    for one_annos_list in annos_these_list:  # 打开第一个结节数据["名字",x,y,z,diam]
        print("one_annos_list:",one_annos_list)  # 打印出第一个结节数据["名字",x,y,z,diam]，此时是世界坐标
        w_center = [one_annos_list[1],one_annos_list[2],one_annos_list[3]]  # w_center为世界坐标的xyz
        print("世界坐标的   结节中心（x，y，z） ",w_center)
        v_center = list(abs(w_center - np.array(origin)))  # /np.array(spacing)  像素间隔为[1,1,1],因此不用再除   abs是绝对值，因为有的为负
        print("图像坐标的   结节中心（x，y，z） ",v_center)  # v_center为图像坐标的xyz
        if zheng is False:    # 如果是反的，由于图反过来了，结节坐标也要反过来
            v_center = [img.shape[2] - 1 - v_center[0],img.shape[1] - 1 - v_center[1],v_center[2]]  # img.shape[2]就是x的总长，-1是因为从0开始，-v_center[0]是减去x的坐标，也即翻转，  z轴不需翻转
        diam = one_annos_list[4]  # 直径
        print("结节直径",diam)
        one_annos = []
        one_annos.append(v_center[0])  # 图像坐标x
        one_annos.append(v_center[1])  # 图像坐标y
        one_annos.append(v_center[2])  # 图像坐标z
        one_annos.append(diam/2)  # 半径
        return_list.append(one_annos)  # 收集这个结节到return_list
        print("one_annos:",one_annos,"[坐标(x,y,z)]")
    return return_list  # 返回该 img_name 的所有结节 [[结节1],[结节2],...]

def bbox_annos_():  # 产生bbox_annos文件，处理所有图的坐标转换
    c = np.array(pd.read_csv(annos_csv))  # c为将annotations读取为数组
    d = []  # 准备放坐标转换后的 ["名字",[[结节1],[结节2],...]]
    for i in range(10):  # 默认你10个subset都下完了
        file_list = os.listdir(luna_path + fengefu+"subset%d" % i)  # 打开D:\datasets\LUNA16\subset0  遍历10个subset
        for ii in file_list:  # 遍历如 subset0 内所有文件
            if len(ii.split(".m")) == 2:  # 如果文件名是mhd文件的话
                name = ii.split(".m")[0]  # 取出文件名，去掉后缀，得到图名
                ct_image_path = find_mhd_path(name)  # 把文件名拿去找对应的mhd文件的绝对地址
                numpyImage, origin, spacing, fanzhuan = read_data(ct_image_path)  # 读取这个mhd文件
                one_annos = get_raw_label(name, numpyImage, c, origin, fanzhuan)  # 进行坐标变换
                d.append([name,one_annos])  # 把变换后的 ["名字",[[结节1],[结节2],...]] 添加到d里
    bbox_annos = pd.DataFrame(d)  # 把 d 转换成excel文件
    bbox_annos.to_excel(new_bbox_annos_path)  # 保存到new_bbox_annos_path


bbox_annos_()


def name(xml_path):  # 从xml文件中取得name
    child = "ResponseHeader"  # 响应 头
    child_child = "SeriesInstanceUid"  # 案例名
    child_child1 = "CTSeriesInstanceUid"  # 案例名，之所以有两个是因为dataset标注不规范
    dom = xml.dom.minidom.parse(xml_path)  # 读取xml文件
    root = dom.documentElement  # 取得树根
    a = root.getElementsByTagName(child)  # 取得树根下的child（ResponseHeader）点
    child_node = a[0].getElementsByTagName(child_child)  # 取得树根下的child下的child_child（SeriesInstanceUid）点
    if child_node==[]:  # 如果值为0
        child_node = a[0].getElementsByTagName(child_child1)  # 取得树根下的child下的child_child1（CTSeriesInstanceUid）点
    child_value = child_node[0].childNodes[0].nodeValue  # 取得该点的值，也就是name
    return child_value  # name

def find_xml_path(name1):
    list1 = []
    for file_list in os.listdir(xml_file_path):  # 遍历xml_file_path文件夹下所有文件
        print(file_list)  # 打印进度
        for ii in os.listdir(xml_file_path + fengefu+file_list):  # 取得xml_file_path文件夹下文件的列表，如157  185 ...
            aim_path = xml_file_path + fengefu+ file_list + fengefu+ii  # 取得xml_file_path文件夹下的  157文件夹下的  文件，如158  159 ..
            with open(aim_path) as f:  # 打开这个文件
                if name(f) == name1:  # 取得这个文件的文件名，如果与输入文件名相符：
                    path = xml_file_path + fengefu+ file_list + fengefu + ii  # 保留这个文件的绝对地址为path（爷找到了）
                    list1.append(path)    # 把这个绝对地址装到 list1 里去
                    print(path)  # 打印绝对地址
        if list1 !=[]:
            return list1  # 得到一个装着绝对地址的列表     if的位置设置为：如果在这个文件夹下找到了，下个文件夹就不找了

def find_mhd_path(name1):
    for file_list in os.listdir(luna_path):  # 遍历luna16文件夹下所有文件
        if file_list.find("subset") != -1:  # 在有subset的文件夹下查找  这一句是为了避免找到seg-lungs-LUNA16文件夹里边去
            for ii in os.listdir(luna_path + fengefu+ file_list):  # 打开luna16文件夹下的文件夹 如subset0，遍历文件
                if len(ii.split(".m")) >1:  # 如果文件中有".m"字符，len就会为2，也即 len > 1
                    if ii.split((".m"))[0] == name1:  # 如果文件名去掉".mhd"后与输入的案例名name一致
                        path = luna_path + fengefu+ file_list + fengefu+ ii  # 取得该文件的绝对地址
                        print(path)
                        return path

# one_name = "1.3.6.1.4.1.14519.5.2.1.6279.6001.287966244644280690737019247886"
# find_xml_path(one_name)
# find_mhd_path(one_name)

def point(xml_path,origin2):  # 需要xml_path 和 图像的原点坐标origin的z轴坐标
    a = []  # 该案例图的 所有z轴和该z轴上点位的列表  [  [z1,[[x1,y1],[x2,y2],...]],  [z2,[[x1,y1],[x2,y2],...]],  ...]
    dom = xml.dom.minidom.parse(xml_path)  # 读取xml文件
    root = dom.documentElement  # 取得树根
    nodeid = root.getElementsByTagName("readingSession")  # 取得树根下的（readingSession）点
    for u in nodeid:  # 遍历所有readingSession点
        child = u.getElementsByTagName("unblindedReadNodule")  # 取得该（readingSession）下的 （unblindedReadNodule）点
        for i in child:   # 遍历该（unblindedReadNodule）下的所有点
            id = i.getElementsByTagName("noduleID")  # 取得该（unblindedReadNodule）下的 （noduleID）点
            id1 = id[0].childNodes[0].nodeValue  # 取得该（unblindedReadNodule）下的 （noduleID）点的值
            if id1:  # 如果（noduleID）的值不为空
                one_all_iou = i.getElementsByTagName("roi")  # 取得该（unblindedReadNodule）下的 （noduleID）下的（roi）点
                for r in one_all_iou:  # 遍历这些（roi）点
                    z = r.getElementsByTagName("imageZposition")  # 取得该（unblindedReadNodule）下的 （noduleID）下的（roi）下的（imageZposition）
                    z1 = float(z[0].childNodes[0].nodeValue)-origin2  # 取得图像坐标的z轴  ， 即 z轴坐标-原点的z轴坐标。  其中 图像在x、y轴不需要变化，在标注时就是按照图像坐标标注的
                    ioux = r.getElementsByTagName("xCoord")  # 取得该z轴切片上的("xCoord")
                    iouy = r.getElementsByTagName("yCoord")  # 取得该z轴切片上的("yCoord")

                    ioux1 = np.array([int(k.childNodes[0].nodeValue) for k in ioux])  # 取得该z轴切片上的x点位（所有x点位），并排列成数组
                    iouy1 = np.array([int(l.childNodes[0].nodeValue) for l in iouy])  # 取得该z轴切片上的y点位（所有y点位），并排列成数组
                    iou = np.array([ioux1,iouy1])  # 数组合并，得到[[x1,x2,...],[y1,y2,...]]
                    point1 = np.transpose(iou)  # 数组转置，得到[[x1,y1],[x2,y2],...]
                    a.append([z1,point1])  # [z轴,z轴对应的点位数组[[x1,y1],[x2,y2],...]]  添加到a列表中
    return a  # 返回该案例图的 所有z轴和z轴上点位 的列表  [  [z1,[[x1,y1],[x2,y2],...]],  [z2,[[x1,y1],[x2,y2],...]],  ...]


def read_data(mhd_file):  # 读取图像数据（包括图，坐标原点，像素间隔，是否需要翻转）
    with open(mhd_file) as f:
        mhd_data = f.readlines()
        for i in mhd_data:  # 判断是否反转，其中 TransformMatrix = 1 0 0 0 1 0 0 0 1\n  代表反转为正True
            if i.startswith('TransformMatrix'):  # 取得以'TransformMatrix'开头的这一行
                zheng = i.split(' = ')[1]  # 取得' = '后边的字符串
                if zheng == '1 0 0 0 1 0 0 0 1\n':  # 如果与'1 0 0 0 1 0 0 0 1\n'相符，其中  100代表x，010代表y，001代表z
                    zheng = True  # 代表是正的，不需要反转
    itkimage = sitk.ReadImage(mhd_file)    # 读取mhd文件
    numpyImage = sitk.GetArrayFromImage(itkimage)   # 从mhd读取到raw，也就是图
    print("读取数据，读取的图片大小（zyx）：",numpyImage.shape)  # 深 depth  *  宽 width  *  高 height
    origin = itkimage.GetOrigin()  # 从mhd读取到origin，也就是原点坐标
    print("读取数据，读取的坐标原点（xyz）：",origin)  # 坐标原点   x,y,z
    spacing = itkimage.GetSpacing()  # 从mhd读取到spacing，也就是像素间隔
    print("读取数据，读取的像素间隔（xyz）：",spacing)  # 像素间隔   x,y,z
    return numpyImage,origin,spacing,zheng


def for_one_(name,wrong):  # 一个处理每张图的函数，输入名字＋一个空列表， 输出 上色好的mask + mhd的绝对地址用于后续存放mask用 + 出错的图的名字

    xml_path_list = find_xml_path(name)  # 根据名字，得到了对应的 xml文件的绝对地址
    ct_image_path = find_mhd_path(name)  # 根据名字，得到了对应的 mhd文件的绝对地址

    ct_image,origin,spacing,fanzhuan = read_data(ct_image_path)  # 根据 mhd文件的绝对地址 ，得到 图，原点信息，像素间隔，是否需要翻转
    s = ct_image.shape  # 拿到 图的尺寸，用来画 全0的mask
    mm  = np.zeros((s[0],s[1],s[2]), dtype=np.int32)   # mm为 全0的mask ， 注意 图.shape 是zyx的，所以顺序不用变
    #取截面  描点
    for i in xml_path_list:  # 取得xml文件的绝对地址，
        list1 = point(i,origin[2])  # 在这个绝对地址内获取所有[ [ z层 ， 点 ]，[ z层 ，点 ] ， ... ]
        print(len(list1))  # 共多少层
        for ii in list1:  # 遍历所有层
            ceng = ii[0]  # ceng为z轴坐标
            print("ceng",ceng)  # 打印层
            pts = ii[1]  # 该层的所有点位  [[x1,y1],[x2,y2],...]
            color = 1  # (0, 255, 0)
    # 解释一下， int（ceng/spacing[2]-1） 是因为ceng代表图像坐标的y轴位置，比如4，是代表4mm，而不是第4层.spacing[2]是z轴的像素间隔，也即每spacing[2]的距离有一层。层数是从0算起，所以-1。这样做的好处是处理后与原图保持一致。
            mm[int(ceng/spacing[2]-1),:,:] = cv.drawContours(mm[int(ceng/spacing[2]-1),:,:], [pts], -1, color=color, thickness=-1)  # 取出这一层，开始染色填充
            mm[int(ceng/spacing[2]-1),:,:] = scipy.ndimage.binary_fill_holes(mm[int(ceng/spacing[2]-1),:,:], structure=None, output=None, origin=0)  # 补洞
    if (mm==np.zeros((s[0],s[1],s[2]), dtype=np.int32)).all():  # 如果没染上色，即仍是全0数组：
        wrong.append(name)  # 认为有错，把名字添加到wrong里
    return mm,ct_image_path,wrong  # 返回染色好的mask，mhd的绝对地址，错误列表

# one_name = "1.3.6.1.4.1.14519.5.2.1.6279.6001.287966244644280690737019247886"
# a,b,c = for_one_(one_name,wrong=[])
# print("a",a.shape,"b",b,"c",c)

# 可视化验证
def plot_2d(image,z = 132):
    # z,y,x#查看第100张图像
    plt.figure()
    plt.imshow(image[z, :, :])
    plt.show()
# z = 240
# plot_2d(a,z=int(z/2.5))
# ct_image_path = find_mhd_path(one_name)
# ct_image,origin,spacing,fanzhuan = read_data(ct_image_path)
# plot_2d(ct_image,z = int(z/2.5))

# 此时，我们拥有处理单个图片mask的能力，处理好的mask与原图保持一致。为了做数据预处理，我们需要把原图和标签 均经过重采样，仅对原图做归一化和去均值

def resample(imgs, spacing, new_spacing=[1,1,1]):  # 重采样 ，即把原图的像素间隔统一
    ###   重采样,坐标原点位置为0
    if len(imgs.shape)==3:   # 如果是3维的话：
        new_shape = []  # 新图大小会变，如  原有4个切片，像素间隔为2.5，重采样后有10个切片，像素间隔为1
        for i in range(3):  # 对每个维度 0，1，2  → z，y，x
            print("（zyx）像素间隔",i,":",spacing[-i-1])   # spacing原顺序为（xyz），spacing[-i-1]顺序为（zyx）
            new_zyx = np.round(imgs.shape[i]*spacing[-i-1]/new_spacing[-i-1])  # round为四舍五入（原图尺寸 * 原像素间隔/新像素间隔）
            new_shape.append(new_zyx)  # new_shape集齐新zyx尺寸
        print("（zyx）新图大小：",new_shape)
        resize_factor = []  # 新图尺寸/原图尺寸   即缩放比例，如  原像素间隔为2.5，新像素间隔为1，放缩比例为1/2.5
        for i in range(3):  # 依次为 0 1 2 → z y x
            resize_zyx = new_shape[i]/imgs.shape[i]  # 放缩比例
            resize_factor.append(resize_zyx)  # 放缩比例 存入 resize_factor ，zoom函数要用
        imgs = zoom(imgs, resize_factor, mode = 'nearest')   # 放缩，边缘使用最近邻，插值默认为三线性插值
        return imgs
    else:
        raise ValueError('wrong shape')  # 本代码只能处理3维数据

