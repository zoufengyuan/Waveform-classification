# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:22:24 2020

@author: FengY Z
"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from tqdm import tqdm
import biosppy
import os
import warnings
import pyentrp.entropy as ent
from scipy.stats import skew, kurtosis
from scipy.signal import resample
from sklearn.preprocessing import minmax_scale, scale
from config import config
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def find_rpeak_ecg(signal_array,fs):
    tmp_lead = biosppy.ecg.ecg(signal_array,show = False,sampling_rate = fs)
    rpeaks = tmp_lead['rpeaks']
    return rpeaks

def find_rpeak_self(signal_array):
    sum_result = 0
    tmp_mean = signal_array[0]
    tmp_rate = 1
    mean_rate_limit = 0
    second_rate_limit = 0
    all_rate = []
    for i,number in enumerate(signal_array):
        sum_result += number
        mean_result = sum_result/(i+1)
        mean_rate = (mean_result-tmp_mean)/tmp_mean
        second_rate = (mean_rate-tmp_rate)/abs(tmp_rate)
        if second_rate>0:
            second_rate_limit+=second_rate
            mean_rate_limit+=mean_rate
        else:
            all_rate.append(second_rate_limit*mean_rate_limit)
            second_rate_limit = 0
            mean_rate_limit = 0
        tmp_mean = mean_result
        tmp_rate = mean_rate
    #得到峰值候选值列表candi_list和其对应的索引candi_index
    sum_result = 0
    tmp_mean = signal_array[0]
    tmp_rate = 1
    tmp_list = []
    mean_rate_limit = 0
    second_rate_limit = 0
    candi_list = []
    candi_index = []
    for i,number in enumerate(signal_array):
        sum_result += number
        mean_result = sum_result/(i+1)
        mean_rate = (mean_result-tmp_mean)/tmp_mean
        second_rate = (mean_rate-tmp_rate)/abs(tmp_rate)
        if second_rate>0:
            second_rate_limit+=second_rate
            mean_rate_limit+=mean_rate
        else:
            if second_rate_limit*mean_rate_limit>np.percentile(all_rate,90):
                candi_list.append(signal_array[i-1])
                candi_index.append(i-1)
            second_rate_limit = 0
            mean_rate_limit = 0
        tmp_mean = mean_result
        tmp_rate = mean_rate
        tmp_list.append(second_rate)
    # In[]
    #运用kmeans进行聚类得到最后的峰值
    kmeans = KMeans(n_clusters=2, random_state=10).fit(np.array(candi_list).reshape((-1,1)))
    result_df = pd.DataFrame({'index':candi_index,'max_peak':candi_list,'label':kmeans.labels_})
    peak_1 = result_df['max_peak'][result_df['label']==0].mean()
    peak_2 = result_df['max_peak'][result_df['label']==1].mean()
    if peak_1>peak_2:
        peak_result = result_df[result_df['label']==0].reset_index()
    elif peak_1<=peak_2:
        peak_result = result_df[result_df['label']==1].reset_index()
    return peak_result['index'].to_list()

def split_cycle(signal_array,fs):
    try:
        ecg_rpeaks = find_rpeak_ecg(signal_array,fs)
    except:
        ecg_rpeaks = [0]
    try:
        self_rpeaks = find_rpeak_self(signal_array)
    except:
        self_rpeaks = [0]
    if len(ecg_rpeaks)>=len(self_rpeaks):
        rpeaks_index = list(ecg_rpeaks)
    else:
        rpeaks_index = self_rpeaks
    n = len(signal_array)
    rpeaks_index = [-1]+rpeaks_index+[n-1]
    cycle_seg = [[rpeaks_index[i]+1,rpeaks_index[i+1]+1] for i in range(len(rpeaks_index)-1)]
    return cycle_seg


'''
diff_list = []
amount_list = []
for array in tqdm(data):
    cycle_seg,rpeak_diff_min,cycle_amount = split_cycle(array,200)
    diff_list.append(rpeak_diff_min)
    amount_list.append(cycle_amount)

info_df = pd.DataFrame({'diff':diff_list,'amount':amount_list})
'''
def signal_resample(signal_array,begin_index,end_index,resample_num):
    array = signal_array[begin_index:end_index]
    resampled_array = resample(array,resample_num)
    return resampled_array

def signal_reshape(signal_array,cycle_seg,cycle_amount = 15,resample_num = 60):
    '''
    cycle_amount表示抽取多少个周期,初步设定为15
    resample_num表示每个周期的长度，初步设定为60
    '''
    reshaped_array = np.zeros((cycle_amount,resample_num))
    seg_n = len(cycle_seg)
    if seg_n > cycle_amount:
        cycle_seg = cycle_seg[:cycle_amount]
    elif seg_n<cycle_amount:
        seg_index = np.arange(seg_n)
        add_seg_index = list(np.random.choice(seg_index,cycle_amount-seg_n,replace = True))
        add_seg = [cycle_seg[i] for i in add_seg_index]
        cycle_seg = cycle_seg + add_seg
    for i in range(cycle_amount):
        resampled_seg = signal_resample(signal_array,cycle_seg[i][0],cycle_seg[i][1],resample_num)
        reshaped_array[i] = resampled_seg
    return reshaped_array

def main_reshape(data,cycle_amount = 15,resample_num = 60):
    n = len(data)
    main_reshaped_data = np.zeros((n,cycle_amount,resample_num))
    for i,array in tqdm(enumerate(data)):
        cycle_seg = split_cycle(array,200)
        reshaped_array = signal_reshape(array,cycle_seg)
        main_reshaped_data[i] = reshaped_array
    return main_reshaped_data
if __name__ == '__main__':
    source_data = pd.read_csv(config.source_data_last)
    data = source_data[config.wave_features].values
    main_reshaped_data = main_reshape(data)    


    
    



























