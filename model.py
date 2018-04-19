# -*- coding: utf-8 -*-
# @Time    : 2018/4/11 18:13
# @Author  : Barry
# @Email   : wcf.barry@foxmail.com
# @File    : model.py
# @Software: PyCharm

import torch
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np



class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(1, 32, 7, stride=2, padding=3)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(64)
        # Sequential 是连续操作的写法
        self.convs = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   )
        self.out_layers = nn.Sequential(nn.Linear(128 * 8 * 8, 1024),
                                        nn.BatchNorm1d(1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 256),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(),
                                        nn.Linear(256, 100),
                                        nn.BatchNorm1d(100),
                                        nn.ReLU(),
                                        )


    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))   # 卷积 BN ReLU
        x = self.pool(x)                        # 池化
        x = F.relu(self.norm2(self.conv2(x)))  # 卷积 BN ReLU
        x = F.relu(self.norm3(self.conv3(x)))  # 卷积 BN ReLU
        x = self.pool(x)
        x = self.convs(x)                      # 连续操作，里面是 conv -> BN -> ReLU -> conv -> BN -> ReLU
        x = self.pool(x)
        x = x.view(-1, 128 * 8 * 8)             # 将图像拉直为向量
        x = self.drop(x)
        x = self.out_layers(x)
        return x

import os
def save_checkpoint(state, save_adress='model_save'):
    name = 'model_parameters.pth.tar'

    folder = os.path.exists(save_adress)
    if not folder:
        os.mkdir(save_adress)
        print('--- create a new folder ---')
    fulladress = save_adress + '\\' + name
    torch.save(state, fulladress)
    print('model saved:', fulladress)

def load_checkpoint(save_adress='model_save'):
    name = 'model_parameters.pth.tar'
    fulladress = save_adress + '\\' + name
    return torch.load(fulladress)
