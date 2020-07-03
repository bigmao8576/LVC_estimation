import tensorflow as tf
import pandas as pd
import os
import pickle
import utils
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
import post_processing as pp
from datetime import datetime

from CRNN import train_for_val,train_for_test

#HYPERPARAMETER SETTING
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

TOTAL_EPOCH = 1000
CNN_CH = 64
RNN_CH = 64
FC_CH =64
FILTER_SIZE = 9
POOLING_SIZE = 4
BATCH_SIZE = 50
L2_LAMBDA = 0.001
LR = 0.0001


excel_path = 'Dan_data_available_for_LVC.xlsx'
test_path = 'all_Hdata.xlsx'
prefix = 'ervinNIH'
test_prefix = 'H_DownSampled Signals'

for OERT_RATIO in [item/10 for item in range(4,6)]:
    

    
    save_path = 'crnn_overratio_%1.1f_'%OERT_RATIO+utils.time_for_saving()
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    #get the original data, including segmenting signals,forming the labels
    
    if os.path.exists(excel_path):
        df=pd.read_excel(excel_path,dtype={'Name':str, 'Value':float})
        
    else:
        raise FileNotFoundError('Please contact the authorts for the dataset')
        
    recording_list = df.to_dict('index')
    recording_list,max_len, max_sig_len = utils.pre_processing(recording_list)
    
    fold_data = utils.get_fold_info(recording_list, OERT_RATIO,experiment = 'seq')
    fold_results = {}
    
    for fold_number in range(10):
        
        model_para = {'over_ratio': OERT_RATIO,
                  'total_epoch': TOTAL_EPOCH,
                  'CNN_channel': CNN_CH,
                  'RNN_channel': RNN_CH,
                  'filter_size': FILTER_SIZE,
                  'pooling_size': POOLING_SIZE,
                  'batch_size': BATCH_SIZE,
                  'fc_channel':FC_CH,
                  'l2_c':L2_LAMBDA,
                  'lr':LR             
            }
        
    
        temp_fold_results = train_for_val(fold_number,fold_data,model_para,save_path)
        
        fold_results['fold_%d'%fold_number] =temp_fold_results   
        
    val_summary = pp.val_perf_seq(fold_results)    # need
      
    # now testing, retrain the model 
    # get the testing and training data
    if os.path.exists(test_path):
        df_test=pd.read_excel(test_path,dtype={'Name':str, 'Value':float})
        
    else:
        raise FileNotFoundError('Please contact the authorts for the dataset')
    
    test_list = df_test.to_dict('index')
    test_list = utils.test_pre_processing(test_list,max_len,max_sig_len)
    
    model_para = {'over_ratio': OERT_RATIO,
                  'total_epoch': TOTAL_EPOCH,
                  'CNN_channel': CNN_CH,
                  'RNN_channel': RNN_CH,
                  'filter_size': FILTER_SIZE,
                  'pooling_size': POOLING_SIZE,
                  'batch_size': BATCH_SIZE,
                  'fc_channel':FC_CH,
                  'l2_c':L2_LAMBDA,
                  'lr':LR             
            }
    
    test_fold_data = utils.get_test_fold_info(test_list,OERT_RATIO,experiment = 'seq')
    
    temp_test_results = train_for_test(test_fold_data,fold_data,model_para,save_path)
    
    
    test_summary = pp.test_perf_seq(temp_test_results) 
    
    Total_results={'fold_result':fold_results,
                   'test_result':temp_test_results,
                   'val_summary':val_summary,
                   'test_summary':test_summary,
                   'model_para':model_para
                   }
    
    pp.write_results(Total_results,save_path)
    pickle.dump( Total_results, open(os.path.join(save_path,'total_results.pkl'), "wb" ) )