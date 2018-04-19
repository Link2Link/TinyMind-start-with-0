# -*- coding: utf-8 -*-
# @Time    : 2018/4/10 17:08
# @Author  : Barry
# @Email   : wcf.barry@foxmail.com
# @File    : data.py
# @Software: PyCharm

import sys
import os
import numpy as np
import torch
import torch.utils.data as data
import cv2
from PIL import Image
from tqdm import tqdm

trainpath = 'train\\'
testpath = 'test1\\'
words = os.listdir(trainpath)   # 按时间排序 从早到晚
category_number = len(words)

img_size = (128, 128)

def loadOneWord(order):
    path = trainpath + words[order] + '\\'
    files = os.listdir(path)
    datas = []
    for file in files:
        file = path + file
        img = np.asarray(Image.open(file))
        img = cv2.resize(img, img_size)
        datas.append(img)
    datas = np.array(datas)
    labels = np.zeros([len(datas), len(words)], dtype=np.uint8)
    labels[:, order] = 1
    return datas, labels

def transData():
    num = len(words)
    datas = np.array([], dtype=np.uint8)
    datas.shape = -1, 128, 128
    labels = np.array([], dtype=np.uint8)
    labels.shape = -1, 100
    for k in tqdm(range(num)):
        data, label = loadOneWord(k)

        datas = np.append(datas, data, axis=0)
        labels = np.append(labels, label, axis=0)

    np.save('data.npy', datas)
    np.save('label.npy', labels)

class TrainSet(data.Dataset):
    def __init__(self, eval=False):
        datas = np.load('data.npy')
        labels = np.load('label.npy')
        index = np.arange(0, len(datas), 1, dtype=np.int)
        np.random.seed(123)
        np.random.shuffle(index)
        if eval:
            index = index[:int(len(datas) * 0.1)]
        else:
            index = index[int(len(datas) * 0.1):]
        self.data = datas[index]
        self.label = labels[index]
        np.random.seed()

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), \
               torch.from_numpy(self.label[index])

    def __len__(self):
        return len(self.data)

def loadtestdata():
    files = os.listdir(testpath)
    datas = []
    for file in tqdm(files):
        file = testpath + file
        img = np.asarray(Image.open(file))
        img = cv2.resize(img, img_size)
        datas.append(img)
    datas = np.array(datas)
    return datas

if __name__ == '__main__':
    transData()
    # datas = np.load('data.npy')
    # labels = np.load('label.npy')
    # index = np.arange(0, len(datas), 1, dtype=np.int)
    # print(datas.shape, labels.shape)
