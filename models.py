#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)



class Alexnet(nn.Module):
    def __init__(self, args):
        super(Alexnet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(args.num_channels, 32, 5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2, stride=1),
        )

        self.conv3 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75),
            nn.Conv2d(64, 96, 3, 1, 1),
        )

        self.conv4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, padding=1, stride=1),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5, True),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5, True),
            nn.Linear(128, 10)
        )

    # 定义前向传播过程，输入为x
    def forward(self, x):
        B = x.shape[0]
        # print("B的值是", B)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)

        c3 = self.conv3(c2)

        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        c5 = c5.view(B, -1)

        exit = self.classifier(c5)

        return exit


# 定义网络结构
class Branchy_Alexnet(nn.Module):
    def __init__(self, args):
        super(Branchy_Alexnet, self).__init__()

        def conv_dw(inp, oup, kernel, stride, padding):
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(args.num_channels, 32, 3, padding=2, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )

        self.conv2 = nn.Sequential(
            # nn.Conv2d(32, 64, 5, padding=2, stride=1),
            conv_dw(32, 64, 3, 1, 2),
        )

        self.branch1_con = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            conv_dw(64, 16, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.branch1_fc = nn.Linear(64, 10)

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            conv_dw(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            conv_dw(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            # nn.Conv2d(96, 64, 3, padding=1, stride=1),
            conv_dw(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            # nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5, True),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5, True),
            nn.Linear(128, 10)
        )

    # 定义前向传播过程，输入为x
    def forward(self, x):
        B = x.shape[0]
        # print("B的值是", B)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)

        # 第一个分支
        b1_c = self.branch1_con(c2)
        b1_c = b1_c.view(B, -1)
        b_exit1 = self.branch1_fc(b1_c)

        c3 = self.conv3(c2)

        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        c5 = c5.view(B, -1)

        exit = self.classifier(c5)

        return b_exit1, exit


class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squzee_channel):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True)
        )

        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, squzee_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(squzee_channel, int(out_channel / 2), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)

        return x


# 定义网络结构
class LB_Net(nn.Module):
    """mobile net with simple bypass"""

    def __init__(self, args):
        super(LB_Net, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(args.num_channels, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fire2 = Fire(96, 128, 16)
        self.fire3 = Fire(128, 128, 16)
        self.fire4 = Fire(128, 256, 32)
        self.fire5 = Fire(256, 256, 32)
        self.fire6 = Fire(256, 384, 48)
        self.fire7 = Fire(384, 384, 48)
        self.fire8 = Fire(384, 512, 64)
        self.fire9 = Fire(512, 512, 64)

        self.conv3 = nn.Conv2d(128, 10, 1)

        self.conv10 = nn.Conv2d(512, 10, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.stem(x)

        f2 = self.fire2(x)
        f3 = self.fire3(f2)

        branch = self.conv3(f3)
        branch = self.avg(branch)
        branch = branch.view(branch.size(0), -1)

        f4 = self.fire4(f3)
        f4 = self.maxpool(f4)
        f5 = self.fire5(f4)
        f6 = self.fire6(f5)
        f7 = self.fire7(f6)
        f8 = self.fire8(f7)
        f8 = self.maxpool(f8)

        f9 = self.fire9(f8)
        c10 = self.conv10(f9)

        x = self.avg(c10)
        x = x.view(x.size(0), -1)

        return F.log_softmax(branch, dim=1), F.log_softmax(x, dim=1)
        #return F.log_softmax(x, dim=1)
        #return branch, x