# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:02:09 2020

@author: FengY Z
"""
import pandas as pd
from config import config
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    source_data = pd.read_csv(config.source_data)
    
    source_data = source_data.drop(['Unnamed: 0','0','1','2','3','sub_id'],axis = 1)
    
    source_data['id'] = np.arange(len(source_data))
    
    source_data.rename(columns = {'Gender(Number)':'Gender'},inplace = True)
    
    #source_data.to_csv(config.source_data_last,index = None)
    
    wave_data = source_data[config.wave_features+['label']]
    
    label_1_index = list(wave_data[wave_data['label']==1].index)
    label_0_index = list(wave_data[wave_data['label']==0].index)
    
    sample_1 = np.random.choice(label_1_index,1)
    sample_0 = np.random.choice(label_0_index,1)
    
    
    wave_1 = wave_data.loc[sample_1,config.wave_features].values.flatten()
    wave_0 = wave_data.loc[sample_0,config.wave_features].values.flatten()
    axis_x = [i for i in range(len(wave_0))]
    fig = plt.figure()
    plt.plot(axis_x,wave_1,label = 1,color = 'r')
    plt.plot(axis_x,wave_0,label = 0,color = 'g')
    plt.legend()
    plt.show()
    








