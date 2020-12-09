# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:49:07 2020

@author: FengY Z
"""

import torch
import numpy as np
import pandas as pd
from config import config
np.random.seed(0)


def split_data(id_list,val_ratio = 0.1):
    n = len(id_list)
    train_id = np.random.choice(id_list,int(n*(1-val_ratio)),replace = False)
    val_id = set(id_list).difference(set(train_id))
    dd = {'train':train_id,
          'val':list(val_id)}
    torch.save(dd,config.train_data)
    

if __name__ == '__main__':
    source_data = pd.read_csv(config.source_data_last)
    '''
    id_0 = source_data['id'][source_data['label'] == 0].to_list()
    id_1 = source_data['id'][source_data['label'] == 1].to_list()
    n_0 = len(id_0)
    chosed_id_1 = list(np.random.choice(id_1,n_0,replace = False))
    id_list = id_0 + chosed_id_1
    '''
    id_list = list(source_data['id'])
    split_data(id_list)


    