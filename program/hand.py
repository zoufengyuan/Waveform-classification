# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:49:11 2020

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
warnings.filterwarnings('ignore')
import shap


def handle_feature(wave_data,fs,resample_num):
    tmp_array = []
    id_list = wave_data.pop('id').to_list()
    for i in tqdm(range(len(wave_data))):
        tmp_list = []
        single_wave = wave_data.iloc[i].values
        try:
            tmp_lead = biosppy.ecg.ecg(single_wave,show = False,sampling_rate = fs)
        except:
            pass
        rpeaks = tmp_lead['rpeaks']
        if rpeaks.shape[0] != 0:
            crest = single_wave[rpeaks]
            rr_intervals = np.diff(rpeaks)#一阶
            drr = np.diff(rr_intervals)#二阶
            r_density = (rr_intervals.shape[0] + 1) / single_wave.shape[0] * fs
            pnn50 = drr[drr >= fs * 0.05].shape[0] / rr_intervals.shape[0]
            rmssd = np.sqrt(np.mean(drr * drr))
            samp_entrp = ent.sample_entropy(rr_intervals, 2,0.2 * np.std(rr_intervals))
            samp_entrp[np.isnan(samp_entrp)] = -2
            samp_entrp[np.isinf(samp_entrp)] = -1
            tmp_list.extend(
                    [crest.mean(),crest.std(),crest.max(),crest.min(),rr_intervals.min(), rr_intervals.max(),
                     rr_intervals.mean(), rr_intervals.std(),
                     skew(rr_intervals), kurtosis(rr_intervals), r_density,
                     pnn50, rmssd, samp_entrp[0], samp_entrp[1]])
        else:
            tmp_list.extend([np.nan] * 15)
        heart_rate = tmp_lead['heart_rate']
        if heart_rate.shape[0] != 0:
            tmp_list.extend([heart_rate.min(), heart_rate.max(),
                             heart_rate.mean(), heart_rate.std(),
                             skew(heart_rate), kurtosis(heart_rate)])
        else:
            tmp_list.extend([np.nan] * 6)
        templates = tmp_lead['templates']
        templates_min = templates.min(axis=0)
        templates_max = templates.max(axis=0)
        templates_diff = templates_max - templates_min
        templates_mean = templates.mean(axis=0)
        templates_std = templates.std(axis=0)
        for j in [templates_diff, templates_mean, templates_std]:
            tmp_rmp = resample(j, num=resample_num)
            tmp_list.extend(list(tmp_rmp))
        tmp_array.append(tmp_list)
    tmp_df = pd.DataFrame(tmp_array)
    tmp_df['id'] = id_list
    return tmp_df

      
if __name__ == '__main__':
    
    source_data = pd.read_csv(config.source_data_last)
    wave_data = source_data[['id']+config.wave_features]
    
    handle_feature_df = handle_feature(wave_data,200,50)
    handle_feature_df['label'] = source_data['label']
    handle_feature_df.to_csv('handle_feature_with_label.csv', index=False, encoding='utf-8')
    
    '''
    import lightgbm as lgb
    from sklearn.model_selection import  train_test_split
    from sklearn.metrics import roc_auc_score,accuracy_score
    
    model_data = handle_feature(wave_data,200,50)
    model_data['label'] = source_data['label']
    
    label = model_data.pop('label')
    
    
    x_train,x_test,y_train,y_test=train_test_split(model_data,label,test_size=0.3,shuffle=True)
    
    train_data=lgb.Dataset(x_train,label=y_train)
    validation_data=lgb.Dataset(x_test,label=y_test)
    params={'boosting_type':'gbdt',
        'learning_rate':0.15,
        'lambda_l1':0.1,
        'lambda_l2':0.2,
        'max_depth':8,
        'objective':'binary'
    }
    clf=lgb.train(params,train_data,valid_sets=[validation_data],num_boost_round  = 300)
    
    y_pred=clf.predict(x_test)
    y_pred_result=[1 if x >0.5 else 0 for x in y_pred]
    print(accuracy_score(y_test,y_pred_result))
    
    module_explainer = shap.KernelExplainer(clf.predict,x_train,nsamples = 5)
    #module_explainer = shap.LinearExplainer(rbf_svc,train_x)
    module_shap_values = module_explainer.shap_values(x_test[:5])
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    #shap.summary_plot(module_shap_values, plot_type="bar")
    shap.summary_plot(module_shap_values, x_test[:5], plot_type="violin")
    plt.tight_layout()
    plt.show()
    '''
    
    
    






















