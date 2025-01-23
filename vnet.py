####    构建vnet


import torch
import torch.nn as nn
import vnet_def
import torch.nn.functional as f





class VNet(nn.Module):
    def __init__(self,num_classes=2):
        super(VNet, self).__init__()

        self.layer0 = vnet_def.res_block(1, 16,"forward0")

        self.layer11 = vnet_def.res_block(16,32,"deconv")
        self.layer2 = vnet_def.res_block(32,32,"forward2")

        self.layer22 = vnet_def.res_block(32,64,"deconv")
        self.layer3 = vnet_def.res_block(64,64,"forward3")

        self.layer33 = vnet_def.res_block(64,128,"deconv",dropout=True)
        self.layer4 = vnet_def.res_block(128,128,"forward3")

        self.layer44 = vnet_def.res_block(128,256,"deconv",dropout=True)
        self.layer5 = vnet_def.res_block(256,256,"forward3")

        self.layer55 = vnet_def.res_block(256,128,"upconv")
        self.layer6 = vnet_def.res_block(256,256,"forward3")

        self.layer66 = vnet_def.res_block(256,64,"upconv")
        self.layer7 = vnet_def.res_block(128, 128,"forward3")

        self.layer77 = vnet_def.res_block(128, 32,"upconv")
        self.layer8 = vnet_def.res_block(64, 64,"forward2")

        self.layer88 = vnet_def.res_block(64, 16,"upconv")
        self.layer9 = vnet_def.res_block(32, 32,"forward1")

        self.layer10 = vnet_def.res_block(32,num_classes,"forward10")
        self.softmax = nn.Softmax(dim=1)   #log_softmax

        self.dropv = nn.Dropout3d()


        ####  提取特征

    def forward(self,x):
        out = self.layer0(x)
        link1 = out  # 16
        out = self.layer11(out)#.deconv(out)
        out = self.layer2(out)#.forward2(out)
        link2 = out  # 32
        out = self.layer22(out)#.deconv(out)
        out = self.layer3(out)#.forward3(out)
        link3 = out  #64
        out = self.layer33(out)#,dropout=True)#.deconv(out)
        out = self.layer4(out)#.forward3(out)
        link4 = out  # 128
        out = self.layer44(out)#,dropout=True)#.deconv(out)
        out = self.layer5(out)#.forward3(out)

        out = self.layer55(out)#.upconv(out)
        out = torch.cat((self.dropv(link4),out),1)
        out = self.layer6(out)#.forward3(out)

        out = self.layer66(out)#.upconv(out)
        out = torch.cat((self.dropv(link3),out),1)
        out = self.layer7(out)#.forward3(out)

        out = self.layer77(out)#.upconv(out)
        out = torch.cat((self.dropv(link2), out), 1)
        out = self.layer8(out)#.forward2(out)

        out = self.layer88(out)#.upconv(out)
        out = torch.cat((self.dropv(link1), out), 1)
        out = self.layer9(out)#.forward1(out)

        out = self.layer10(out)#.pointconv(out)
        out = self.softmax(out)
        return out

