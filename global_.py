from pathlib import Path
import torch

# 修改区域
luna_path = r"G:\datasets\LUNA16"
xml_file_path = r'G:\datasets\LIDC-IDRI\LIDC-XML-only\tcia-lidc-xml'
annos_csv = r'G:\datasets\LUNA16\CSVFILES\annotations.csv'
new_bbox_annos_path = r"G:\datasets\sk_output\bbox_annos\bbox_annos.xlsx"
mask_path = r'G:\datasets\LUNA16\seg-lungs-LUNA16'
output_path = r"G:\datasets\sk_output"
bbox_img_path = r"G:\datasets\sk_output\bbox_image"
bbox_msk_path = r"G:\datasets\sk_output\bbox_mask"
wrong_img_path = r"G:\datasets\wrong_img.xlsx"
zhibiao_path = r'G:\datasets\sk_output\zhibiao'
model_path = r'G:\datasets\sk_output\model'
msg_path = r'G:\datasets\sk_output\msgs.xlsx'

# 训练设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 没gpu就用cpu
valid_epoch_each = 5  # 每几轮验证一次

# 建立文件夹结构
Path(new_bbox_annos_path).parent.mkdir(exist_ok=True, parents=True)
Path(bbox_img_path).mkdir(exist_ok=True, parents=True)
Path(bbox_msk_path).mkdir(exist_ok=True, parents=True)
Path(model_path).mkdir(exist_ok=True, parents=True)
Path(zhibiao_path).mkdir(exist_ok=True, parents=True)
