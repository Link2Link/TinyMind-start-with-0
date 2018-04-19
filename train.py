# -*- coding: utf-8 -*-
# @Time    : 2018/4/10 17:07
# @Author  : Barry
# @Email   : wcf.barry@foxmail.com
# @File    : train.py
# @Software: PyCharm

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import data
import model as model
from tqdm import tqdm

n_epoch, batch_size = 10, 512

trainset = data.TrainSet(eval=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
evalset = data.TrainSet(eval=True)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle=True)

net = model.net()
if torch.cuda.is_available():
    net.cuda()
criterion = nn.CrossEntropyLoss()


def train(epoch):
    net.train() # 网络处于训练模式，会导致dropout启用
    correct = 0
    sum = 0
    T = 0

    for batch_index, (datas, labels) in enumerate(trainloader, 0):
        labels = labels.max(1)[1]
        datas = Variable(datas).float()
        datas = datas.view(-1, 1, 128, 128)
        labels = Variable(labels).long()
        if torch.cuda.is_available():
            datas = datas.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        outputs = net(datas)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        T += 1
        pred_choice = outputs.data.max(1)[1]
        correct += pred_choice.eq(labels.data).cpu().sum()
        sum += len(labels)

        print('batch_index: [%d/%d]' % (batch_index, len(trainloader)),
              'Train epoch: [%d]' % (epoch),
              'correct/sum:%d/%d, %.4f' % (correct, sum, correct / sum))

def eval(epoch):
    net.eval()  # 弯网络处于测试模式，dropout停用，BN放射变换停止
    correct = 0
    sum = 0
    for batch_index, (datas, labels) in enumerate(evalloader, 0):
        labels = labels.max(1)[1]
        datas = Variable(datas).cuda().float()
        datas = datas.view(-1, 1, 128, 128)
        labels = Variable(labels).cuda().long()
        # optimizer.zero_grad()
        outputs = net(datas)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        pred_choice = outputs.data.max(1)[1]
        correct += pred_choice.eq(labels.data).cpu().sum()
        sum += len(labels)
        print('batch_index: [%d/%d]' % (batch_index, len(evalloader)),
              'Eval epoch: [%d]' % (epoch),
              'correct/sum:%d/%d, %.4f' % (correct, sum, correct / sum))

if __name__ == '__main__':
    # 是否装载模型参数
    load = False

    if load:
        checkpoint = model.load_checkpoint()
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    # 设置优化器
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)
    # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=1e-1, weight_decay=1e-4)

    for epoch in range(start_epoch, n_epoch):
        train(epoch)

        # 保存参数
        checkpoint = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
        model.save_checkpoint(checkpoint)

        eval(epoch)
