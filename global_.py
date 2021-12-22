# 修改区域

luna_path = r"D:\datasets\LUNA16"
xml_file_path = r'D:\datasets\LIDC-IDRI\LIDC-XML-only\tcia-lidc-xml'
annos_csv = r'D:\datasets\LUNA16\CSVFILES\annotations.csv'
new_bbox_annos_path = r"D:\datasets\sk_output\bbox_annos\bbox_annos.xls"
mask_path = r'D:\datasets\LUNA16\seg-lungs-LUNA16'
output_path = r"D:\datasets\sk_output"
bbox_img_path = r"D:\datasets\sk_output\bbox_image"
bbox_msk_path = r"D:\datasets\sk_output\bbox_mask"
wrong_img_path = r"D:\datasets\wrong_img.xls"
zhibiao_path = r'D:\datasets\sk_output\zhibiao'
model_path = r'D:\datasets\sk_output\model'


xitong = "windows"  # "linux"


# 训练设置
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 没gpu就用cpu
valid_epoch_each = 5  # 每几轮验证一次

if xitong == "linux":
    fengefu = r"/"
else:
    fengefu = r"\\"




