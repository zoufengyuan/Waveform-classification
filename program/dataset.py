# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:43:50 2020

@author: FengY Z
"""

import os
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal


def resample(sig, target_point_num=None):
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return np.array(sig).reshape(-1,1)


def scaling(X, sigma=0.05):
    scalingFactor = np.random.normal(loc=1.0,
                                     scale=sigma,
                                     size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def shift(sig, interval=10):
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset
    return sig


def transform(sig, train=False):#加噪音
    sig = resample(sig, config.target_point_num)
    np.random.seed()
    if np.random.rand() > 0.5: sig = scaling(sig)
    if np.random.rand() > 0.5: sig = shift(sig)
    sig = sig.transpose()#将(2500,8)转化成(8,2500)
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig


class ECGDataset(Dataset):
    def __init__(self, data_path,train=True):
        super(ECGDataset, self).__init__()
        dd = torch.load(data_path)
        self.train = train
        self.id = dd['train'] if train else dd['val']
        source_data = pd.read_csv(config.source_data_last)
        self.data = source_data.loc[self.id,config.wave_features].values
        self.label = source_data.loc[self.id,config.label].values

    def __getitem__(self, index):
        df = self.data[index]
        x = transform(df, self.train).reshape((1, 1, config.target_point_num))
        target = self.label[index]
        target = torch.tensor(target, dtype=torch.float32)
        return x, target

    def __len__(self):
        return len(self.id)


if __name__ == '__main__':
    d = ECGDataset(config.train_data)
    #print(d[0])