from global_annos import *
from global_ import *
import torch
import torch.nn as nn
import torch.utils.data
import os
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt


class dice_loss(nn.Module):  # dice损失，做反向传播
    def __init__(self,c_num=2):  # 格式需要
        super(dice_loss, self).__init__()
    def forward(self,data,label):  # 格式需要
        n = data.size(0)  # data.size(0)指 batch_size 的值，也就是一个批次几个
        dice_list = []  # 用来放本批次中的每一个图的dice
        all_dice = 0.  # 一会 算本批次的平均dice 用
        for i in range(n):  # 本批次内，拿一个图出来

            my_label11 = label[i]  # my_label11为取得的对应label，也可以说是前景为结节的分割图
            my_label1 = torch.abs(1 - my_label11)  # my_label1为 前景为非结节的分割图   1-1=0，1-0=1，这样就互换了

            my_data1 = data[i][0]  # my_data1为我的模型预测出的 前景为非结节的分割图
            my_data11 = data[i][1]  # my_data11为我的模型预测出的 前景为结节的分割图

            m1 = my_data1.view(-1)  # 把my_data1拉成一维       ps：前景为非结节的分割图
            m2 = my_label1.view(-1)  # 把my_label1拉成一维     ps：前景为非结节的分割图

            m11 = my_data11.view(-1)  # 把my_data1拉成一维     ps：前景为结节的分割图
            m22 = my_label11.view(-1)  # 把my_label1拉成一维   ps：前景为结节的分割图

            dice = 0  # dice初始化为0
            dice += (1-(( 2. * (m1 * m2).sum() +1 ) / (m1.sum() + m2.sum() +1)))  # dice loss = 1-DSC的公式，比较的是 前景为非结节的分割图
            dice += (1-(( 2. * (m11 * m22).sum() + 1) / ( m11.sum()+m22.sum()+ 1)))  # dice loss = 1-DSC的公式，比较的是 前景为结节的分割图
            dice_list.append(dice)  # 里面放本批次中的所有图的dice，每张图的dice为 前景结节 和 前景非结节 两图的dice loss 求和


        for i in range(n):  # 遍历本批次所有图
            all_dice += dice_list[i]  # 求和
        dice_loss = all_dice/n

        return dice_loss  # 返回本批次所有图的平均dice loss

Loss = dice_loss().to(DEVICE)  # 损失函数布置到gpu或cpu上

def train_model(model, device, train_loader, optimizer, epoch):  # 训练模型
    # 模型训练-----调取方法
    model.train()  # 用来训练的
    loss_need = []  # 记录loss
    tqdr = tqdm(enumerate(train_loader))  # 用一下tqdm函数，也就是进度条工具（枚举）
    for batch_index, (data, target) in tqdr:  # 取batch索引，（data，target），也就是图和标签
        data, target = data.to(device), target.to(device)  # 放到gpu或cpu上
        output = model(data)  # 图 进模型 得到预测输出
        loss = Loss(output, target)  # 计算损失
        optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器走一步
        train_loss = loss.item()  # 取得损失值
        loss_need.append(train_loss)  # 放到loss_need列表里
        tqdr.set_description("Train Epoch : {} \t train Loss : {:.6f} ".format(epoch, loss.item()))  # 实时显示损失
    train_loss = np.mean(loss_need)  # 求平均
    print("train_loss", train_loss)  # 打印平均损失
    return train_loss,loss_need  # 返回平均损失，损失列表

def test_model(model, device, test_loader, epoch,test):    # 加了个test  1是想打印时好看（区分valid和test）  2是test要打印图，需要特别设计
    # 模型训练-----调取方法
    model.eval()  # 用来验证或测试的
    test_loss = 0.0  # 测试损失
    PA = IOU = DICE = P =R =F1 = 0  # 指标归0
    tqrr = tqdm(enumerate(test_loader))  # 进度条
    with torch.no_grad():  # 不进行 梯度计算（反向传播）
        for batch_index,(data, target) in tqrr:  # 枚举batch索引，（图，标签）
            if test:  # 如果是测试，做可视化；如果是验证，则不做
                data_cpu = data.clone().cpu()  # 取出图到cpu
                my_label_cpu = target.clone().cpu()  # 取出预测的二值分割到cpu
                for i in range(len(data_cpu)):  # 取出改batch中的单张图
                    true_img_tensor = data_cpu[i][0]  # 取图得到张量tensor，注意这里的[0]是因为我们在dataset部分给图增加了一个维度
                    true_label_tensor = my_label_cpu[i]  # 取得预测的二值分割张量tensor
                    use_plot_2d(true_img_tensor,true_label_tensor,z=8,batch_index=batch_index,i=i,true_label=True)  # 存图，这里存标签图到pic

            data, target = data.to(device), target.to(device)
            torch.cuda.empty_cache()
            output = model(data)    #(output.shape) torch.Size([4, 2, 96, 96, 96])
            loss = Loss(output, target)#*nllLoss(out1, target)
            test_loss += loss.item()

            PA0, IOU0, DICE0, P0, R0, F10,tn, fp, fn, tp = zhibiao(output, target)
            PA += PA0
            IOU += IOU0
            DICE += DICE0
            P += P0
            R += R0
            F1 += F10
            if test:
                name = 'Test'
            else:
                name = 'Valid'
            tqrr.set_description("{} Epoch : {} \t {} Loss : {:.6f} \t tn, fp, fn, tp:  {:.0f}  {:.0f}  {:.0f}  {:.0f} ".format(name,epoch,name, loss.item(),tn, fp, fn, tp))
            if test:
                data_cpu = data.clone().cpu()
                my_output_cpu = output.clone().cpu()
                for i in range(len(data_cpu)):
                    img_tensor = data_cpu[i][0]  # 96 * 96 * 96
                    label_tensor = torch.gt(my_output_cpu[i][1], my_output_cpu[i][0])  # 96 * 96 * 96
                    use_plot_2d(img_tensor,label_tensor,z=8,batch_index=batch_index,i=i)

        test_loss /= len(test_loader)
        PA /= len(test_loader)
        IOU /= len(test_loader)
        DICE /= len(test_loader)
        P /= len(test_loader)
        R /= len(test_loader)
        F1 /= len(test_loader)

        print(" Epoch : {} \t {} Loss : {:.6f} \t DICE :{:.6f} PA: {:.6f} ".format(epoch, name,test_loss,DICE,PA))

        return test_loss, [PA, IOU, DICE, P, R, F1]






class myDataset(Dataset):

    def __init__(self, data_path, label_path):   ###  transform 我没写
        self.data = self.get_img_label(data_path)   ## 图的位置列表
        self.label = self.get_img_label(label_path)   ## 标签的位置列表

        self.annos_img = self.get_annos_label(self.data)  # 图的位置列表 输入进去  吐出  结节附近的图的【【图片位置，结节中心，半径】列表】
        self.annos_label = self.get_annos_label(self.label)    #112


    def __getitem__(self, index):
        img_all = self.annos_img[index]
        label_all = self.annos_label[index]
        img = np.load(img_all[0])    # 载入的是图片地址
        label = np.load(label_all[0])    # 载入的是label地址
        cut_list = []      ##  切割需要用的数

        for i in range(len(img.shape)):   ###  0,1,2   →  z,y,x
            if i == 0:
                a = img_all[1][-i - 1] - 8  ### z
                b = img_all[1][-i - 1] + 8
            else:
                a = img_all[1][-i-1]-48   ### z
                b = img_all[1][-i-1]+48   ###
            if a<0:
                if i == 0:
                    a = 0
                    b = 96
                else:
                    a = 0
                    b = 96
            elif b>img.shape[i]:
                if i == 0 :
                    a = img.shape[i] - 16
                    b = img.shape[i]
                else:
                    a = img.shape[i]-96
                    b = img.shape[i]
            else:
                pass

            cut_list.append(a)
            cut_list.append(b)


        img = img[cut_list[0]:cut_list[1],cut_list[2]:cut_list[3],cut_list[4]:cut_list[5]]   ###  z,y,x
        label = label[cut_list[0]:cut_list[1],cut_list[2]:cut_list[3],cut_list[4]:cut_list[5]]   ###  z,y,x

        # plot_3d(img)
        # plot_3d(label)
        img = np.expand_dims(img,0)  ##(1, 96, 96, 96)
        img = torch.tensor(img)
        img = img.type(torch.FloatTensor)
        label = torch.Tensor(label).long()  ##(96, 96, 96) label不用升通道维度
        torch.cuda.empty_cache()
        return img,label    ### 从这里出去还是96*96*96


    def __len__(self):
        return len(self.annos_img)


    @staticmethod
    def get_img_label(data_path):   ###  list 地址下所有图片的绝对地址

        img_path = []
        for t in data_path:  ###  打开subset0，打开subset1
            data_img_list = os.listdir(t)  ## 列出图
            img_path += [os.path.join(t, j) for j in data_img_list]  ##'/public/home/menjingru/dataset/sk_output/bbox_image/subset1/1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217.npy'
        img_path.sort()
        return img_path  ##返回的也就是图像路径 或 标签路径

    @staticmethod
    def get_annos_label(img_path):
        annos_path = []  # 这里边要装图的地址，结节的中心，结节的半径    要小于96/4 # ###半径最大才12

        ### ok   ,   anoos 是处理好的列表了，我只需要把他们对比一下是否在列表里，然后根据列表里的坐标输出一个列表  就可以了   在__getitem__里边把它切下来就行

        for u in img_path:  # 图的路径
            if xitong== "windows":
                name = '1'+u.split(r"\1")[-1].split(".np")[0]  # 拿到图的名字
            else:
                name = u.split(r"/")[-1].split(".np")[0]  # 拿到图的名字
            for one in annos_list:  # 遍历有结节的图
                if one[0] == name:  # 如果有结节的图的名字 == 输入的图的名字
                    for l in range(len(one[1])):  # 数一数有几个结节
                        annos_path.append(
                            [u, [one[1][l][0], one[1][l][1], one[1][l][2]], one[1][l][3]])  # 图的地址，结节的中心，结节的半径
        return annos_path  # ###半径最大才12







def zhibiao(data,label):   #   data  n,2,96,96,96  label  n,96,96,96

    ###        这里需要把data变换成label形式，方法是取大为1

    n = data.size(0)
    PA, IOU, DICE, P, R, F1 ,TN, FP, FN, TP= 0,0,0,0,0,0,0,0,0,0


    for i in range(n):

        empty_data = torch.gt(data[i][1], data[i][0])
        empty_data = empty_data.long()  #pred label

        my_data = empty_data  ##  得到处理好的 pred label（96*96*96）
        my_label = label[i]   ##      标准答案     label


        my_data = my_data.cpu().numpy()
        my_data = numpy_list(my_data)
        # print(my_data)

        my_label = my_label.cpu().numpy()
        my_label = numpy_list(my_label)


        confuse = confusion_matrix(my_label,my_data,labels=[0,1])  ### 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(my_label,my_data, labels=[0,1]).ravel()
        all = tn + fp + fn + tp
        # print("tn, fp, fn, tp",tn, fp, fn, tp)
        diag = torch.diag(torch.from_numpy(confuse))
        b = 0
        for ii in diag:
            b += ii
        diag = b

        PA += float(torch.true_divide(diag , all ))  ##  混淆矩阵  对角线/总数
        # IOU += float(torch.true_divide(diag,(2 * all - diag)))    ##  交并比
        # DICE += float(2 * torch.true_divide(diag,2 * all))
        IOU += float(torch.true_divide(tp,tp+fp+fn))    ##  交并比
        # DICE += float(2 * torch.true_divide(diag,2 * all))
        DICE += float(torch.true_divide(2*tp,fp+fn+2*tp))
        if tp + fp ==0:
            P += tp/(tp + fp+1)    ## 精准率  （注意不是精度）
        else:
            P += tp/(tp + fp)    ## 精准率  （注意不是精度）

        if tp + fn == 0:
            R += tp/(tp + fn+1)    ## 召回率
        else:
            R += tp/(tp + fn)    ## 召回率

        # if P + R == 0:
        #     F1 += 2 * P * R / (P + R+1)
        # else:
        #     F1 += 2 * P * R / (P + R)

        TN += tn
        FP += fp
        FN += fn
        TP += tp
    TN /= n
    FP /= n
    FN /= n
    TP /= n

    PA = PA/n
    IOU = IOU/n
    DICE = DICE/n
    P = P/n
    R = R/n
    if P + R == 0:
        F1 += 2 * P * R / (P + R + 1)
    else:
        F1 += 2 * P * R / (P + R)
    return PA,IOU,DICE,P,R,F1,TN, FP, FN, TP



def numpy_list(numpy):
    x = []
    numpy_to_list(x,numpy)
    return x


def numpy_to_list(x,numpy):
    for i in range(len(numpy)):
        if type(numpy[i]) is np.ndarray:
            numpy_to_list(x,numpy[i])
        else:
            x.append(numpy[i])








def show_loss(loss_list,STR,path):  ###  损失列表，损失名称，保存位置
    EPOCH = len(loss_list)  ##  训练集中是  总epoch   验证集中是  总epoch/每多少epoch进行验证集的epoch数   测试集中就一个数不用画
    x1 = range(0, EPOCH)
    y1 = loss_list

    plt.plot(x1, y1, "-" ,label=STR)
    plt.legend()

    plt.savefig(path +'/%s.jpg'%STR)
    plt.close()


def use_plot_2d(image,output,z = 132,batch_index=0,i=0,true_label=False):
    # z,y,x#查看第100张图像
    plt.figure()
    p = image[z, :, :] +0.25 ## 96*96     这是归一化后的
    p = torch.unsqueeze(p,dim=2)
    q = output[z, :, :]  ##96*96
    q = (q * 0.2).float()
    q = torch.unsqueeze(q,dim=2)
    q = p + q
    q[q >1] = 1
    r = p
    cat_pic = torch.cat([r,q,p],dim=2)  #  红色为空，my_label为绿色，原图为蓝色
    plt.imshow(cat_pic)

    path = zhibiao_path       #  我真的懒得引入参数了，这个path 就是 zhibiao_path
    if true_label:
        if not os.path.exists(path +fengefu+'true_pic'):  # 建立subset文件夹
            os.mkdir(path +fengefu+'true_pic')
        plt.savefig(path +'/true_pic/%d_%d.jpg'%(batch_index,i))
    else:
        if not os.path.exists(path +fengefu+'pic'):  # 建立subset文件夹
            os.mkdir(path +fengefu+'pic')
        plt.savefig(path +'/pic/%d_%d.jpg'%(batch_index,i))
    plt.close()
