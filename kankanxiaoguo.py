
import torch.utils.data
import torch.optim as optim
from train_def import *
# from vnet import DSCVNet
import pandas as pd
import time
from global_ import *

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    ###   不知道为啥!!!4核都没报错！！！
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#

BATCH_SIZE = 4
EPOCH = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


torch.cuda.empty_cache()





data_test_path = []  ### 测试用
label_test_path = []
for ii in range(6,7):  ### 8,9   测试集
    data_test_path.append('/home/zhangfuchun/menjingru/dataset/sk_output/bbox_image/subset%d' % ii)
    label_test_path.append('/home/zhangfuchun/menjingru/dataset/sk_output/bbox_mask/subset%d' % ii)
dataset_test = myDataset(data_test_path, label_test_path)
test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=BATCH_SIZE, shuffle=False,
                                              num_workers=16)
print("Test_dataloader_ok")



test_start = time.perf_counter()
torch.cuda.empty_cache()
test_loss_list = []
test_zhibiao_list = []


model = torch.load(r'/home/zhangfuchun/menjingru/dataset/sk_output/model/best_model.pth')   ###在这里改

test_loss, test_zhibiao = test_model(model, DEVICE, test_loader,EPOCH,test=True)
test_loss_list.append(test_loss)     ###  得到测试的损失loss
test_zhibiao_list.append(test_zhibiao)

test_loss_pd = pd.DataFrame(test_loss_list)
test_loss_pd.to_excel(zhibiao_path + "/测试损失.xls")
test_zhibiao_pd = pd.DataFrame(test_zhibiao_list)
test_zhibiao_pd.to_excel(zhibiao_path + "/测试验证指标[PA, IOU, DICE, P, R, F1].xls")


test_end = time.perf_counter()
test_time =test_end-test_start
print('Running time: %s Seconds'%test_time)
test_time_list = []
test_time_list.append(test_time)
test_time_pd = pd.DataFrame(test_time_list)
test_time_pd.to_excel(zhibiao_path + "/测试时间.xls")
