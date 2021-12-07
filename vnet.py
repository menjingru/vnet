####    构建vnet


import torch
import torch.nn as nn
import vnet_def
import torch.nn.functional as f

#
#
# class VNet(nn.Module):
#     def __init__(self,num_classes=2):
#         super(VNet, self).__init__()
#
#         self.layer1 = vnet_def.res_block(1, 16,"forward1")
#         self.layer11 = vnet_def.res_block(16,16,"deconv")
#         self.layer2 = vnet_def.res_block(16,16,"forward2")
#         self.layer22 = vnet_def.res_block(16,32,"deconv")
#         self.layer3 = vnet_def.res_block(32,32,"forward3")
#         self.layer33 = vnet_def.res_block(32,64,"deconv")
#         self.layer4 = vnet_def.res_block(64,64,"forward3")
#         self.layer44 = vnet_def.res_block(64,128,"deconv")
#         self.layer5 = vnet_def.res_block(128,128,"forward3")
#         self.layer55 = vnet_def.res_block(128,256,"upconv")
#         self.layer6 = vnet_def.res_block(64+256,256,"forward3")
#         self.layer66 = vnet_def.res_block(256,128,"upconv")
#         self.layer7 = vnet_def.res_block(32 + 128, 128,"forward3")
#         self.layer77 = vnet_def.res_block(128, 64,"upconv")
#         self.layer8 = vnet_def.res_block(16 + 64, 64,"forward2")
#         self.layer88 = vnet_def.res_block(64, 32,"upconv")
#         self.layer9 = vnet_def.res_block(16 + 32, 32,"forward1")
#         self.layer10 = vnet_def.res_block(32,num_classes,"pointconv")   ###   num_classes=2
#         self.softmax = f.softmax
#
#
#         ####  提取特征
#
#     def forward(self,x):
#         out = self.layer1(x,"0")#.forward1(out)
#         link1 = out
#         out = self.layer11(out,"0")#.deconv(out)
#         out = self.layer2(out,"0")#.forward2(out)
#         link2 = out
#         out = self.layer22(out,"0")#.deconv(out)
#         out = self.layer3(out,"0")#.forward3(out)
#         link3 = out
#         out = self.layer33(out,"0")#.deconv(out)
#         out = self.layer4(out,"0")#.forward3(out)
#         link4 = out
#         out = self.layer44(out,"0")#.deconv(out)
#         out = self.layer5(out,"0")#.forward3(out)
#         out = self.layer55(out,"0")#.upconv(out)
#         # out = torch.cat((link4,out),1)   ###按照第一个维度拼接
#         out = self.layer6(out,link4)#.forward3(out)
#         out = self.layer66(out,"0")#.upconv(out)
#         # out = torch.cat((link3,out),1)
#         out = self.layer7(out,link3)#.forward3(out)
#         out = self.layer77(out,"0")#.upconv(out)
#         # out = torch.cat((link2, out), 1)
#         out = self.layer8(out,link2)#.forward2(out)
#         out = self.layer88(out,"0")#.upconv(out)
#         # out = torch.cat((link1, out), 1)
#         out = self.layer9(out,link1)#.forward1(out)
#         out = self.layer10(out,"0")#.pointconv(out)
#         out = self.softmax(out,dim=1)
#
#         return out



class VNet(nn.Module):
    def __init__(self,num_classes=2):
        super(VNet, self).__init__()

        self.layer1 = vnet_def.res_block(1, 16,"forward1")
        self.layer11 = vnet_def.res_block(16,16,"deconv")
        self.layer2 = vnet_def.res_block(16,32,"forward2")
        self.layer22 = vnet_def.res_block(32,32,"deconv")
        self.layer3 = vnet_def.res_block(32,64,"forward3")
        self.layer33 = vnet_def.res_block(64,64,"deconv")
        self.layer4 = vnet_def.res_block(64,128,"forward3")
        self.layer44 = vnet_def.res_block(128,128,"deconv")
        self.layer5 = vnet_def.res_block(128,256,"forward3")
        self.layer55 = vnet_def.res_block(256,256,"upconv")
        self.layer6 = vnet_def.res_block(128+256,256,"forward3")
        self.layer66 = vnet_def.res_block(256,256,"upconv")
        self.layer7 = vnet_def.res_block(64 + 256, 128,"forward3")
        self.layer77 = vnet_def.res_block(128, 128,"upconv")
        self.layer8 = vnet_def.res_block(32 + 128, 64,"forward2")
        self.layer88 = vnet_def.res_block(64, 64,"upconv")
        self.layer9 = vnet_def.res_block(16 + 64, 32,"forward1")
        self.layer10 = vnet_def.res_block(32,num_classes,"pointconv")   ###   num_classes=2
        self.softmax = nn.Softmax(dim=1)   #log_softmax


        ####  提取特征

    def forward(self,x):
        out = self.layer1(x)#.forward1(out)
        link1 = out
        out = self.layer11(out)#.deconv(out)
        out = self.layer2(out)#.forward2(out)
        link2 = out
        out = self.layer22(out)#.deconv(out)
        out = self.layer3(out)#.forward3(out)
        link3 = out
        out = self.layer33(out)#.deconv(out)
        out = self.layer4(out)#.forward3(out)
        link4 = out
        out = self.layer44(out)#.deconv(out)
        out = self.layer5(out)#.forward3(out)
        out = self.layer55(out)#.upconv(out)
        out = torch.cat((link4,out),1)   ###按照第一个维度拼接
        out = self.layer6(out)#.forward3(out)
        out = self.layer66(out)#.upconv(out)
        out = torch.cat((link3,out),1)
        out = self.layer7(out)#.forward3(out)
        out = self.layer77(out)#.upconv(out)
        out = torch.cat((link2, out), 1)
        out = self.layer8(out)#.forward2(out)
        out = self.layer88(out)#.upconv(out)
        out = torch.cat((link1, out), 1)
        out = self.layer9(out)#.forward1(out)
        out = self.layer10(out)#.pointconv(out)
        out = self.softmax(out)
        return out

