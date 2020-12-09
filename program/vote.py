# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:02:03 2020

@author: FengY Z
"""

import pandas as pd
import numpy as np
from config import config
import lightgbm as lgb
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix
import shap
import matplotlib.pyplot as plt


def load_txt(txt_file):
    f = pd.read_csv(txt_file,sep = '\t',header = None)
    feature_name = int(txt_file.split('.')[0])
    f.rename(columns = {0:'id',1:'feature_'+str(feature_name)},inplace = True)
    return f
def load_txt_multi_feature(txt_file):
    f = pd.read_csv(txt_file,sep = '\t',header = None)
    f.rename(columns = {0:'id'},inplace = True)
    return f

def net_feature(feature_num_list):
    feature_txt_list = [str(i)+'.txt' for i in feature_num_list]
    base_df = load_txt(feature_txt_list[0])
    for txt_file in feature_txt_list[1:]:
        df = load_txt(txt_file)
        base_df = base_df.merge(df,on = 'id')
    feature_vars_list = ['feature_'+str(x) for x in feature_num_list]
    base_df['net_feature'] = base_df[feature_vars_list].mean(axis = 1)
    #base_df = base_df.drop(feature_vars_list,axis = 1)
    return base_df

def net_feature_multi(feature_num_list):
    feature_txt_list = [str(i)+'.txt' for i in feature_num_list]
    base_df = load_txt_multi_feature(feature_txt_list[0])
    id_series = base_df.pop('id')
    base_df = base_df.values
    for txt_file in feature_txt_list[1:]:
        df = load_txt_multi_feature(txt_file)
        id_series = df.pop('id')
        df = df.values
        base_df = base_df+df
    base_df = pd.DataFrame(base_df/4)
    base_df['id'] = list(id_series)
    base_df.rename(columns = {i:'feature_'+str(i) for i in range(130)},inplace = True)
    return base_df

def combined_df():
    handle_feature_df = pd.read_csv('handle_feature_with_label.csv')
    base_feature_df = pd.read_csv(config.source_data_last)[['id']+config.base_features]
    net_feature_df = net_feature_multi([1,2,3,4])
    for df in [net_feature_df,handle_feature_df]:
        base_feature_df = base_feature_df.merge(df,on = 'id',how = 'left')
    return base_feature_df

if __name__ == '__main__':
    model_data = combined_df()
    del model_data['id']
    
    label = model_data.pop('label')
    
    x_train,x_test,y_train,y_test=train_test_split(model_data,label,test_size=0.2,shuffle=True)
    
    train_data=lgb.Dataset(x_train,label=y_train)
    validation_data=lgb.Dataset(x_test,label=y_test)
    params={'boosting_type':'gbdt',
        'learning_rate':0.15,
        'lambda_l1':0.7,
        'lambda_l2':0.7,
        'max_depth':15,
        'objective':'binary',
        'feature_fraction':0.4,
        'bagging_fraction':0.4,
        'min_sum_hessian_in_leaf':0.5,
        'min_data_in_leaf':300,
        'early_stopping_round':10,
        'metric':'binary_error',
        'imbalanced':True,
        'max_bin':127
    }
    clf=lgb.train(params,train_data,valid_sets=[validation_data],num_boost_round  = 300)
    y_pred_train=clf.predict(x_train)
    y_pred_train_result=[1 if x >0.5 else 0 for x in y_pred_train]
    print(accuracy_score(y_train,y_pred_train_result))
    print(confusion_matrix(y_train,y_pred_train_result))
    
    y_pred=clf.predict(x_test)
    y_pred_result=[1 if x >0.5 else 0 for x in y_pred]
    print(accuracy_score(y_test,y_pred_result))
    print(confusion_matrix(y_test,y_pred_result))
    print(roc_auc_score(y_test,y_pred))
    
#    module_explainer = shap.KernelExplainer(clf.predict,x_train,nsamples = 10)
#    #module_explainer = shap.LinearExplainer(rbf_svc,train_x)
#    module_shap_values = module_explainer.shap_values(x_test[:20])
#    plt.rcParams['font.sans-serif']=['SimHei']
#    plt.rcParams['axes.unicode_minus'] = False
    #shap.summary_plot(module_shap_values, plot_type="bar")
#    shap.summary_plot(module_shap_values, x_test[:20], plot_type="violin")
#    plt.tight_layout()
#    plt.show()
    x_test_index = x_test.index.to_list()
    base_wave_df = pd.read_csv(config.source_data_last)[config.wave_features]
    test_wave_df = base_wave_df.loc[x_test_index]
    test_wave_df['y_pred'] = y_pred_result
    test_wave_df['label'] = y_test
    
    
    y_pred_label_1_1 = test_wave_df[config.wave_features][(test_wave_df['y_pred']==1)&(test_wave_df['label']==1)]
    y_pred_label_1_0 = test_wave_df[config.wave_features][(test_wave_df['y_pred']==1)&(test_wave_df['label']==0)]
    y_pred_label_0_0 = test_wave_df[config.wave_features][(test_wave_df['y_pred']==0)&(test_wave_df['label']==0)]
    y_pred_label_0_1 = test_wave_df[config.wave_features][(test_wave_df['y_pred']==0)&(test_wave_df['label']==1)]
    
    axis_x = np.arange(1250)
    
    plt.figure()
    plt.plot(axis_x,y_pred_label_1_1.iloc[0].values,label = 'y_pred_label_1_1',color ='r')
    plt.plot(axis_x,y_pred_label_1_0.iloc[0].values,label = 'y_pred_label_1_0',color ='g')
    plt.plot(axis_x,y_pred_label_0_0.iloc[0].values,label = 'y_pred_label_0_0',color ='b')
    plt.plot(axis_x,y_pred_label_0_1.iloc[0].values,label = 'y_pred_label_0_1')
    plt.legend()
    plt.show()
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
