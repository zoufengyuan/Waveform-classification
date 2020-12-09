# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:04:26 2020

@author: FengY Z
"""
import os

class Config:
    data_root = r'..//'
    source_data = os.path.join(data_root,'source_data//hypotensive_forecast_data.csv')
    source_data_last = os.path.join(data_root,'source_data//hypotensive_forecast_data_new.csv')
    wide_data_dir = os.path.join(data_root,'wide_data')
    base_features = ['spo2','HR','resp','expire_flag','values','age','Gender']
    wave_features = [str(x) for x in range(4,1254)]
    label = 'label'
    train_data = os.path.join(wide_data_dir,'train.pth')
    target_point_num = 1000
    current_w = 'current.pth'
    best_w = 'best.pth'
    #target_point_num = 60
    batch_size = 128
    lr = 4e-5
    ckpt = './ckpt'
    model_name = 'after'
    current_w = 'current.pth'
    best_w = 'best.pth'
    sub_dir = './submit'
    max_epoch = 100
    stage_epoch = [24, 48]
    cycle_amount = 15
    resample_num = 60
    lr_decay = 10
    
    
    
    
    
    
    
    

    
config = Config()
