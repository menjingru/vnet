import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽通知和警告信息
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用gpu0


from global_ import *
from global_annos import *
from train_def import *
from vnet import VNet
import time
import torch.utils.data
import torch.optim as optim


BATCH_SIZE = 4  # 2
EPOCH = 200  # 共跑200轮


print(DEVICE)

model = VNet(2)  # 模型
model = model.to(DEVICE)  # 模型部署到gpu或cpu里

torch.cuda.empty_cache()  # 时不时清下内存


######       数据准备

data_path = []  # 装图所在subset的绝对地址，如 [D:\datasets\sk_output\bbox_image\subset0,D:\datasets\sk_output\bbox_image\subset1,..]
label_path = []  # 装标签所在subset的绝对地址，与上一行一致，为对应关系
for i in range(0,8):  # 0,1,2,3,4,5,6,7   训练集
    data_path.append(bbox_img_path+fengefu+'subset%d' % i)  # 放入对应的训练集subset的绝对地址
    label_path.append(bbox_msk_path+fengefu+'subset%d' % i)
dataset_train = myDataset(data_path, label_path)  # 送入dataset
print(len(dataset_train))
train_loader = torch.utils.data.DataLoader(dataset_train,  # 生成dataloader
                                               batch_size=BATCH_SIZE, shuffle=False,
                                               num_workers=0)#16)  # 警告页面文件太小时可改为0
print("train_dataloader_ok")



data_valid_path = []  # 装图所在subset的绝对地址
label_valid_path = []  # 装标签所在subset的绝对地址
for j in range(8,9):  # 8   验证集
    data_valid_path.append(bbox_img_path+fengefu+'subset%d' % j)  # 放入对应的验证集subset的绝对地址
    label_valid_path.append(bbox_msk_path+fengefu+'subset%d' % j)
dataset_valid = myDataset(data_valid_path, label_valid_path)  # 送入dataset
valid_loader = torch.utils.data.DataLoader(dataset_valid,  # 生成dataloader
                                               batch_size=BATCH_SIZE, shuffle=False,
                                               num_workers=0)#16)  # 警告页面文件太小时可改为0
print("valid_dataloader_ok")

data_test_path = []  # 装图所在subset的绝对地址
label_test_path = []  # 装标签所在subset的绝对地址
for ii in range(9,10):  # 9   测试集
    data_test_path.append(bbox_img_path+fengefu+'subset%d' % ii)  # 放入对应的测试集subset的绝对地址
    label_test_path.append(bbox_msk_path+fengefu+'subset%d' % ii)
dataset_test = myDataset(data_test_path, label_test_path)  # 送入dataset
test_loader = torch.utils.data.DataLoader(dataset_test,  # 生成dataloader
                                              batch_size=BATCH_SIZE, shuffle=False,
                                              num_workers=0)#16)  # 警告页面文件太小时可改为0
print("Test_dataloader_ok")


######       数据准备完成，开始训练

start = time.perf_counter()  # 记录训练开始时间

train_loss_list = []  # 用来记录训练损失
valid_loss_list = []  # 用来记录验证损失


minnum = 0  # 寻找最小损失，损失最小意味着模型最佳
mome = 0.99  # 动量，可以认为是前冲的速度
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=mome, weight_decay=1e-8)  # weight_decay质量，认为是前冲的惯性
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1, last_epoch=-1)  # 设置优化器在训练时改变，每3轮lr变为原来的0.1倍,如果中途停止则从头开始

train_loss1 = 0.0
lr = 1e-1
for epoch in range(1, EPOCH + 1):  # 每一个epoch  训练一轮   检测一轮
    if epoch ==180:  # 180轮时动量变为0.9，即更容易落入低点，也更难以回避局部最优点
        mome = 0.9
    train_loss = train_model(model, DEVICE, train_loader, optimizer, epoch)  # 训练
    train_loss1 = train_loss  # 训练损失
    train_loss_list.append(train_loss)  # 记录每个epoch训练损失
    train_loss_pd = pd.DataFrame(train_loss_list)  # 存成excel格式
    train_loss_pd.to_excel(zhibiao_path + "/第%d个epoch的训练损失.xls" %(epoch))

    torch.save(model, model_path+fengefu+'train_model.pth')  # 保存训练模型
    torch.cuda.empty_cache()  # 清理内存

    if epoch%valid_epoch_each == 0:   #  如：每5轮验证一次

        valid_loss, valid_zhibiao = test_model(model, DEVICE, valid_loader,epoch,test=False)   # 验证
        dice1 = valid_zhibiao[2]  # 记录dice值
        valid_loss_list.append(valid_loss)  # 验证损失
        valid_loss_pd = pd.DataFrame(valid_loss_list)  # 存成excel格式
        valid_loss_pd.to_excel(zhibiao_path + "/第%d个epoch的验证损失.xls" % (epoch))

        if epoch == valid_epoch_each:  # 第一此验证，如：epoch==5
            torch.save(model, model_path+fengefu+'best_model.pth')  # 保存为最好模型
            minnum = valid_loss  # 刚开始，令min为该loss
            print("minnum",minnum)  # 打印最小验证损失

        elif valid_loss < minnum:  # 如果验证损失 比 记录中最小的验证损失 更小

            print("valid_loss < minnum",valid_loss, "<", minnum)  # 打印 这一轮验证损失更小，所以准备更新了
            minnum = valid_loss  # 最小验证损失 更新为 这一轮验证损失
            torch.save(model, model_path+fengefu+'best_model.pth')  # 保存为最好模型，这里是直接覆盖了之前的best_model
            zhibiao = valid_zhibiao  # 把指标也记录一下
            zhibiao_pd = pd.DataFrame(zhibiao)  # 存成excel格式
            zhibiao_pd.to_excel(zhibiao_path + "/目前为止最合适的model指标：第%d个epoch的验证指标[PA, IOU, DICE, P, R, F1].xls" % epoch)
        else:
            pass  # 验证损失没有变小则不做处理

        torch.cuda.empty_cache()  # 清理内存
    optimizer.step()  # 重要修改TAT  加了这个学习率才会变
    scheduler.step()  # 重要修改TAT  加了这个学习率才会变
end = time.perf_counter()  # 记录训练结束时间
train_time = end-start  # 记录总耗时
print('Running time: %s Seconds' % train_time)  # 打印总耗时
time_list = list([train_time])  # 总耗时转化为列表
train_time_pd = pd.DataFrame(time_list)  # 存成excel格式
train_time_pd.to_excel(zhibiao_path + "/总epoch的训练时间（不包含测试）.xls")


# 训练和验证 结束，保存的最好模型在 model_path +fengefu +'best_model.pth'，用它进行测试

test_start = time.perf_counter()  # 记录测试开始时间
torch.cuda.empty_cache()  # 清一下内存

test_loss_list = []  # 准备放测试损失
test_zhibiao_list = []  # 准备放测试指标


model = torch.load(model_path +fengefu +'best_model.pth')  # 载入最好模型
model = model.to(DEVICE)  # 部署到gpu或cpu上

test_loss, test_zhibiao = test_model(model, DEVICE, test_loader,EPOCH,test=True)  # 测试
test_loss_list.append(test_loss)  # 测试损失
test_zhibiao_list.append(test_zhibiao)  # 测试指标

test_loss_pd = pd.DataFrame(test_loss_list)  # 存成excel格式
test_loss_pd.to_excel(zhibiao_path + "/测试损失.xls")
test_zhibiao_pd = pd.DataFrame(test_zhibiao_list)  # 存成excel格式
test_zhibiao_pd.to_excel(zhibiao_path + "/测试验证指标[PA, IOU, DICE, P, R, F1].xls")


test_end = time.perf_counter()  # 记录测试结束时间
test_time =test_end-test_start  # 记录测试耗时
print('Running time: %s Seconds' % test_time)  # 打印总耗时
test_time_list = list([test_time])  # 测试时间转化为列表
test_time_pd = pd.DataFrame(test_time_list)  # 存成excel格式
test_time_pd.to_excel(zhibiao_path + "/测试时间.xls")
