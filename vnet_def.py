
#
# import torch.nn as nn
# import numpy as np
# import torch
# import math
# import torch.nn.functional as f
#
# class sk_block(nn.Module):  ###   明天照着这个继续写   https://blog.csdn.net/zahidzqj/article/details/105982058
#     def __init__(self, in_channel, out_channel, M=2, r=16, L=32):  ###   M是分支数，r是降维比率，L是维度下界
#         super(sk_block, self).__init__()
#         self.in_channel = in_channel  ####  我们需要的 输入 要等与  输出
#         self.out_channel = out_channel
#         self.M = M
#         self.r = r
#         self.L = L
#         g = min(in_channel, 16, out_channel)
#         self.k_3_conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1,
#                                   padding=1, groups=g)  # .cuda()
#         self.dilated_conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1,
#                                       padding=2, dilation=2, groups=g)  # .cuda()  # 膨胀卷积
#         self.ave_pooling = nn.AdaptiveAvgPool3d(1)  # .cuda()   #  全局平均池化
#         d = max(out_channel // r, L)
#         self.fc1 = nn.Linear(out_channel, d)  # .cuda()
#         self.fc2 = nn.Linear(d, out_channel)  # .cuda()
#         self.softmax = f.softmax
#         self.prelu = nn.PReLU()
#         self.bn = nn.BatchNorm3d(out_channel)
#         self.bn1 = nn.BatchNorm1d(d)
#         self.point = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
#
#     def forward(self, x):
#         res = self.point(x)
#         out1 = self.k_3_conv(x)  ##  这里 通道数 变了       (BS,C,SHAPE)
#         out1 = self.bn(out1)
#         out1 = self.prelu(out1)
#         out2 = self.dilated_conv(x)
#         out2 = self.bn(out2)
#         out2 = self.prelu(out2)
#         out = out1.add(out2)
#         out1d = self.ave_pooling(out)  ##  （BD,C,1*1*1）
#         out1d = torch.flatten(out1d, start_dim=1)
#         out = self.fc1(out1d)
#         # out = self.bn1(out)
#         out = self.prelu(out)
#         outfc1 = self.fc2(out)
#         #
#         outfc1 = self.prelu(outfc1)
#         outfc2 = self.fc2(out)
#         #
#         outfc2 = self.prelu(outfc2)
#         outfc = torch.cat((outfc1, outfc2), 0)
#
#         out = self.softmax(outfc, 1)  #
#         k_3_out = out[0, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
#         dil_out = out[1, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
#         se1 = torch.mul(k_3_out, out1)  ###   这里两个不同大小的张量要相乘了   先把一个张量扩张一下   再点乘
#         se2 = torch.mul(dil_out, out2)
#         out = se1.add(se2)
#         out = res + out
#         return out  # 有正有负，在0附近
# # 由于数据量少，小网效果更好
#
# class mv2_block(nn.Module):
#     def __init__(self,in_channel,out_channel):
#         super(mv2_block,self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#
#         # self.d_conv = nn.Conv3d(in_channels=2 * in_channel,out_channels=2 * in_channel,kernel_size=5,stride=1,padding=2,groups=2*in_channel)  # 深度卷积
#         self.d_conv = nn.Conv3d(in_channels=2 * in_channel,out_channels=2 * in_channel,kernel_size=3,stride=1,padding=1,groups=2*in_channel)  # 深度卷积
#         # self.d_conv = se_block(2*in_channel, 2 * in_channel)  # 深度卷积
#
#         self.p_conv1 = nn.Conv3d(in_channels=in_channel, out_channels=2 * in_channel, kernel_size=1, stride=1,groups=1)  # 点卷积1
#         # self.p_conv1 = se_block(in_channel, 3 * in_channel)  # 点卷积1
#
#         self.p_conv2 = nn.Conv3d(in_channels=2 * in_channel, out_channels=out_channel, kernel_size=1, stride=1,groups=1)  # 点卷积2
#         # self.p_conv2 = se_block(3 * in_channel, out_channel)
#
#         self.prelu = nn.PReLU()
#
#
#     def forward(self,x):
#         resres = x
#         mv2res = res_block(self.in_channel, self.out_channel, "pointconv")
#         resres = mv2res(resres)
#
#         outupc = self.p_conv1(x)
#         outupc = self.prelu(outupc)
#
#         out = self.d_conv(outupc)      #  分了16组  每组  4，1，96，96，96
#         out = self.prelu(out)
#         out = self.p_conv2(out)
#         ### 线性激活函数我没找到  用 y=x代替咯
#         out = resres.add(out)
#         return out
#
# class se_block(nn.Module):
#     def __init__(self,in_channel,out_channel,r=16,L=4):
#         super(se_block, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.point = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
#         self.k_3_conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1,padding=1)
#         self.bn = nn.BatchNorm3d(out_channel)
#         self.prelu = nn.PReLU()
#         self.ave_pooling = nn.AdaptiveAvgPool3d(1)
#         d = max(out_channel // r,L)
#         self.fc1 = nn.Linear(out_channel, d)  # .cuda()
#         self.fc2 = nn.Linear(d, out_channel)  # .cuda()
#
#     def forward(self,x):
#         res = self.point(x)
#         # res = self.bn(res)
#         res = self.prelu(res)
#
#         out = self.k_3_conv(x)
#         out = self.bn(out)
#         out = self.prelu(out)
#         # print(out.shape)
#         out1d = self.ave_pooling(out)  ##  （BD,C,1*1*1）
#         out1d = torch.flatten(out1d, start_dim=1)
#         # print(out1d.shape)
#         out_mid = self.fc1(out1d)
#         out_mid = self.prelu(out_mid)
#         out_out = self.fc2(out_mid)
#         out_out = self.prelu(out_out)
#         out_out = out_out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         # print(out_out.shape)
#         out = torch.mul(out, out_out)
#         out = out + res
#         return out
#
# class res_block(nn.Module):  ##nn.Module
#     def __init__(self, i_channel, o_channel,lei):
#         super(res_block, self).__init__()
#         self.in_c = i_channel
#         self.out_c = o_channel
#
#         self.conv1 = nn.Conv3d(in_channels=self.in_c, out_channels=self.out_c, kernel_size=5, stride=1,padding=2).cuda()#.to(device)   ###  从 输入channel 到 输出channel
#         self.conv2 = nn.Conv3d(in_channels=self.out_c, out_channels=self.out_c, kernel_size=5, stride=1,padding=2).cuda()#.to(device)   ###  从 输出channel 到 输出channel  （叠加层）
#         # self.conv1 = sk_block(in_channel=i_channel, out_channel=o_channel).cuda()
#         # self.conv2 = sk_block(in_channel=o_channel, out_channel=o_channel).cuda()
#
#         self.conv3 = nn.Conv3d(in_channels=self.out_c, out_channels=self.out_c, kernel_size=2, stride=2).cuda()#.to(device)   ###  卷积下采样
#
#         self.conv4 = nn.ConvTranspose3d(in_channels=self.out_c, out_channels=self.out_c, kernel_size=2, stride=2).cuda()#.to(device)   ###  反卷积上采样
#
#         self.conv5 = nn.Conv3d(in_channels=self.in_c, out_channels=self.out_c, kernel_size=1, stride=1).cuda()#.to(device)   ###  点卷积
#
#         self.bn = nn.BatchNorm3d(o_channel).cuda()#.to(device)
#         self.prelu = nn.PReLU().cuda()#.to(device)
#         self.lei = lei
#
#
#     def forward(self,x):
#         if self.lei == "forward1":
#             # out = self.forward1(x)
#             x = x.to(torch.float32)
#             res = x  ###   记录下输入时的 x
#             res1 = res_block(self.in_c, self.out_c, "pointconv")
#             res = res1(res)
#             out = self.conv1(x)
#             out = self.bn(out)
#             out = res.add(out)
#             out = self.prelu(out)
#         elif self.lei == "forward2":
#             # out = self.forward2(x)
#             res = x  ###   记录下输入时的 x
#             res1 = res_block(self.in_c, self.out_c, "pointconv")
#             res = res1(res)
#             out = self.conv1(x)
#             out = self.bn(out)
#             out = self.prelu(out)
#             out = self.conv2(out)
#             out = self.bn(out)
#
#             out = res.add(out)
#             out = self.prelu(out)
#         elif self.lei == "forward3":
#             # out = self.forward3(x)
#             res = x  ###   记录下输入时的 x
#             res1 = res_block(self.in_c, self.out_c, "pointconv")
#             res = res1(res)
#             out = self.conv1(x)
#             out = self.bn(out)
#             out = self.prelu(out)
#             out = self.conv2(out)
#             out = self.bn(out)
#             out = self.prelu(out)
#             out = self.conv2(out)
#             out = self.bn(out)
#             out = res.add(out)
#             out = self.prelu(out)
#         elif self.lei == "deconv":
#             # out = self.deconv(x)
#             out = self.conv3(x)
#             out = self.bn(out)
#             out = self.prelu(out)
#         elif self.lei == "upconv":
#             # out = self.upconv(x)
#             out = self.conv4(x)
#             out = self.bn(out)
#             out = self.prelu(out)
#         elif self.lei == "pointconv":
#             # out = self.pointconv(x)
#             out = self.conv5(x)
#             out = self.bn(out)
#             out = self.prelu(out)
#         else:
#             print("有问题")
#             out = x
#         return out


import torch.nn as nn
import torch
import torch.nn.functional as f


class res_block(nn.Module):  ##nn.Module
    def __init__(self, i_channel, o_channel,lei):
        super(res_block, self).__init__()
        self.in_c = i_channel
        self.out_c = o_channel

        if self.in_c == 1:
            self.conv1 = nn.Conv3d(in_channels=i_channel, out_channels=o_channel, kernel_size=5, stride=1, padding=2)

        elif self.in_c ==80:
            self.conv1 = nn.Conv3d(in_channels=i_channel, out_channels=o_channel, kernel_size=5, stride=1, padding=2)

        else:

            self.conv1 = nn.Conv3d(in_channels=i_channel, out_channels=i_channel, kernel_size=5, stride=1, padding=2)

        self.conv2 = nn.Conv3d(in_channels=i_channel, out_channels=o_channel, kernel_size=5, stride=1, padding=2)


        self.conv3 = nn.Conv3d(in_channels=o_channel, out_channels=o_channel, kernel_size=2, stride=2).cuda()  ###  卷积下采样

        self.conv4 = nn.ConvTranspose3d(in_channels=o_channel, out_channels=o_channel, kernel_size=2, stride=2).cuda()   ###  反卷积上采样

        self.conv5 = nn.Conv3d(in_channels=i_channel, out_channels=o_channel, kernel_size=1, stride=1).cuda()   ###  点卷积

        self.bn = nn.BatchNorm3d(i_channel).cuda()
        self.bn1 = nn.BatchNorm3d(o_channel).cuda()
        self.prelu = nn.ELU().cuda()
        self.lei = lei
        self.drop = nn.Dropout3d()

    def forward(self,x):
        if self.lei == "forward1":
            out = self.forward1(x)
        elif self.lei == "forward2":
            out = self.forward2(x)
        elif self.lei == "forward3":
            out = self.forward3(x)
        elif self.lei == "deconv":
            out = self.deconv(x)
        elif self.lei == "upconv":
            out = self.upconv(x)
        else:
            out = self.pointconv(x)
        return out




    def forward1(self, x):
        x = x.to(torch.float32)
        res = x   ###   记录下输入时的 x
        res1 = res_block(self.in_c,self.out_c,"pointconv")
        res = res1(res)
        # print(x.shape)           ####记下   torch.Size([1, 1, 192, 160, 160])
        out = self.conv1(x)
        # print(out.shape)         ####记下   torch.Size([1, 16, 192, 160, 160])
        out = self.bn1(out)
        out = self.drop(out)
        out = self.prelu(out)
        out = res.add(out)
        return out

    def forward2(self,x ):
        res = x   ###   记录下输入时的 x
        res1 = res_block(self.in_c, self.out_c, "pointconv")
        res = res1(res)
        out = self.conv1(x)
        out = self.bn(out)
        out = self.drop(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.drop(out)
        out = self.prelu(out)

        out = res.add(out)


        return out

    def forward3(self, x):
        res = x   ###   记录下输入时的 x
        res1 = res_block(self.in_c, self.out_c, "pointconv")
        res = res1(res)
        out = self.conv1(x)
        out = self.bn(out)
        out = self.drop(out)
        out = self.prelu(out)
        out = self.conv1(out)
        out = self.bn(out)
        out = self.drop(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.drop(out)
        out = self.prelu(out)

        out = res.add(out)


        return out

    def deconv(self,x):
        out = self.conv3(x)
        out = self.bn(out)
        out = self.prelu(out)
        return out

    def upconv(self,x):
        out = self.conv4(x)
        out = self.bn(out)
        out = self.prelu(out)
        return out

    def pointconv(self,x):
        out = self.conv5(x)
        out = self.bn1(out)
        out = self.prelu(out)
        return out

