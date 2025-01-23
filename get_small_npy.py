# coding=utf-8
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽通知和警告信息
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用gpu0

from train_def import *
import torch.utils.data

BATCH_SIZE = 1
EPOCH = 1

torch.cuda.empty_cache()  # 时不时清下内存

data_path = []  # 装图所在subset的绝对地址，如 [D:\datasets\sk_output\bbox_image\subset0,D:\datasets\sk_output\bbox_image\subset1,..]
label_path = []  # 装标签所在subset的绝对地址，与上一行一致，为对应关系
for i in range(0,10):  # 0,1,2,3,4,5,6,7   训练集
    data_path.append(str(Path(bbox_img_path)/f'subset{i}'))  # 放入对应的训练集subset的绝对地址
    label_path.append(str(Path(bbox_msk_path)/f'subset{i}'))
dataset_train = cutDataset(data_path, label_path)  # 送入dataset
print(len(dataset_train))
train_loader = torch.utils.data.DataLoader(dataset_train,  # 生成dataloader
                                               batch_size=BATCH_SIZE, shuffle=False,
                                               num_workers=0)#16)  # 警告页面文件太小时可改为0
print("train_dataloader_ok")

all_msg_list = []
for epoch in range(1, EPOCH + 1):  # 每一个epoch  训练一轮   检测一轮
    tqdr = tqdm(enumerate(train_loader))  # 用一下tqdm函数，也就是进度条工具（枚举）

    for batch_index, one_list in tqdr:
        all_msg_list.append([i[0] for i in one_list])
df = pd.DataFrame(all_msg_list, columns=['img_path', 'lbl_path','msg'])  # msg是结节的中心 z,y,x
df.to_excel(msg_path)
