#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 19:50:01 2020

@author: bigmao
"""

from scipy import io
import os
import numpy as np
from scipy import interpolate
import pyprind
from copy import deepcopy
import random
from skimage.util import view_as_windows
import datetime
import matplotlib.pyplot as plt

def read_signal(item,prefix = 'ervinNIH',fs = 4500):
    '''
    original_file is the file path
    start_frame,end_frame are labeled from data_record lists
    
    input: 
        item is a dictionary
        
    output 
        upsampled signal
    '''
    
    if type(item).__name__ != 'dict':
        raise ValueError('the input is not a dict')
        
    
    file_name = item['singal_file_name']
    file_path=os.path.join(prefix,item['patient_id'],file_name+'.mat')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError('the signal file %s cannot be found'%file_path)
    try:
        data_old=io.loadmat(file_path)[file_name]
    except:
        first_two=file_name[0:2].upper()
        file_name=first_two+file_name[2:]
        data_old=io.loadmat(file_path)[file_name]
      
        

    num,_=np.shape(data_old)
    
    t_old=np.array([i/4000 for i in range(num)])
    f = interpolate.interp1d(t_old, data_old.T)
    t_new=np.arange(0, t_old[-1], 1/fs)
    data_new=f(t_new).T
    return data_new

def read_test_signal(item,test_prefix='H_DownSampled Signals',fs = 4500):
    '''
    original_file is the file path
    start_frame,end_frame are labeled from data_record lists
    
    input: 
        item is a dictionary
        
    output 
        upsampled signal
    '''
    
    if type(item).__name__ != 'dict':
        raise ValueError('the input is not a dict')
        
    
    file_name = item['singal_file_name']
    file_path=os.path.join(test_prefix,file_name +'.mat')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError('the signal file %s cannot be found'%file_path)

    data_old=io.loadmat(file_path)[file_name]
    
    num,_=np.shape(data_old)
    t_old=np.array([i/4000 for i in range(num)])
    f = interpolate.interp1d(t_old, data_old.T)
    t_new=np.arange(0, t_old[-1], 1/fs)
    data_new=f(t_new).T
    
    return data_new


def GT_gen(item):
    
    if type(item).__name__ != 'dict':
        raise ValueError('the input is not a dict')
    last_LC_f=item['LVC_offset_30']-1 #last frame is already open
    gt=np.zeros([item['end_frame_30']-item['start_frame_30']+1,3])
    gt[:(item['LVC_onset_30']-item['start_frame_30']),0]=1.0
    gt[(item['LVC_onset_30']-item['start_frame_30']):(last_LC_f-item['start_frame_30']+1),1]=1.0
    gt[(last_LC_f-item['start_frame_30']+1):,2]=1.0
    
    return np.float32(gt)

def seg_signal_point(start_frame,end_frame):
    """
    The strat_frame and end_frame are input,
    output are the corresponding time points in signal
    """
    start_time_point=(start_frame-1)*150
    end_time_point=(end_frame)*150 # slice doesn't consider the last element!!!!!
    return start_time_point,end_time_point

def pre_processing(recording_list):
    print('Now segmenting the training signals')
    sample_num = len(recording_list)
    bar = pyprind.ProgBar(sample_num, monitor=True)
    for ind in recording_list.keys():
    
        item = recording_list[ind]
        
        # read original signal
        item['singal_file_name'] = item['patient_id']+item['file_id']
        
        # get start frame and end frame of the swallow
        
        item['start_frame_30'] = np.int(np.ceil(item['start_frame']/2))
        item['end_frame_30'] = np.int(np.ceil(item['end_frame']/2))
        
        # get the onset and offset of LVC
        item['LVC_onset_30'] = np.int(np.ceil(item['Laryngeal Vestibule Closure']/2))
        item['LVC_offset_30'] = np.int(np.ceil(item['Laryngeal Vestibule Reopening']/2))
        
        # some LVC offset is later than the end of swallow
        if item['end_frame_30'] - item['LVC_offset_30'] <=3:
            item['end_frame_30'] = item['LVC_offset_30']+3
        # get ground truth    
        item['GT_3C'] = GT_gen(item)
        item['GT'] = item['GT_3C'][:,1]
        
        item['label_len'] = item['GT'].shape[0]
    
        # get signal seg
        signal_data = read_signal(item)
        start_frame = item['start_frame_30']
        end_frame = item['end_frame_30']
        start_point, end_point = seg_signal_point(start_frame,end_frame)    
        item['signal_seg'] = np.float32(signal_data[start_point:end_point])
    
        bar.update()
        
    print(bar)
    
    # for mask, padding
    total_len = []
    total_sig_len = []
    
    # get the maximum length of the label and signal
    for ind in recording_list.keys():
        item = recording_list[ind]
        
        total_len.append(item['label_len'])
        total_sig_len.append(item['signal_seg'].shape[0])
    
    max_len = max(total_len)
    max_sig_len = max(total_sig_len)
    
    for ind in recording_list.keys():
        item = recording_list[ind]
        temp_mask = np.zeros([max_len,1], dtype=np.int)
        temp_mask[:item['label_len']] = 1
        item['mask'] = temp_mask
    
        gt_pad = np.zeros([max_len,1], dtype=np.float32)
        gt_pad[:item['label_len'],0] = item['GT']
        
        item['GT_pad'] = gt_pad
        
        temp_signal = deepcopy(item['signal_seg'])
        
        sig_max,sig_min = np.max(temp_signal,0),np.min(temp_signal,0)
        
        norm_sig = (2*(temp_signal-sig_min)/(sig_max-sig_min))-1
        
        sig_pad = np.zeros([max_sig_len,4], dtype=np.float32)
        sig_pad[:item['signal_seg'].shape[0]]=norm_sig
        
        item['sig_pad'] = sig_pad
        
    return recording_list, max_len, max_sig_len

def test_pre_processing(test_list,max_len,max_sig_len):
    print('Now segmenting the testing signals')
    for ind in test_list.keys():
        item = test_list[ind]
        
        item['singal_file_name'] = item['Participant']+item['file_num']
        
        
        # get start frame and end frame of the swallow
        
        item['start_frame_30'] = np.int(np.ceil(item['start_frame']/2))
        item['end_frame_30'] = np.int(np.ceil(item['end_frame']/2))
        
        # get the onset and offset of LVC
        item['LVC_onset_30'] = np.int(np.ceil(item['Laryngeal Vestibule Closure']/2))
        item['LVC_offset_30'] = np.int(np.ceil(item['Laryngeal Vestibule Reopening']/2))
        
        # some LVC offset is later than the end of swallow
        if item['end_frame_30'] - item['LVC_offset_30'] <=3:
            item['end_frame_30'] = item['LVC_offset_30']+3
        
        
        item['GT_3C'] = GT_gen(item)
        item['GT'] = item['GT_3C'][:,1]
        
        item['label_len'] = item['GT'].shape[0]
        
        # get signal seg
        signal_data = read_test_signal(item)
        start_frame = item['start_frame_30']
        end_frame = item['end_frame_30']
        start_point, end_point = seg_signal_point(start_frame,end_frame)    
        item['signal_seg'] = np.float32(signal_data[start_point:end_point])    
        
    
        temp_mask = np.zeros([max_len,1],dtype=np.int)
        temp_mask[:item['label_len']] = 1
        item['mask'] = temp_mask
    
        gt_pad = np.zeros([max_len,1], dtype=np.float32)
        gt_pad[:item['label_len'],0] = item['GT']
        
        item['GT_pad'] = gt_pad
        
        temp_signal = deepcopy(item['signal_seg'])
        
        sig_max,sig_min = np.max(temp_signal,0),np.min(temp_signal,0)
        
        norm_sig = (2*(temp_signal-sig_min)/(sig_max-sig_min))-1
        
        sig_pad = np.zeros([max_sig_len,4], dtype=np.float32)
        sig_pad[:item['signal_seg'].shape[0]]=norm_sig
        
        item['sig_pad'] = sig_pad
    return test_list


def get_test_fold_info(test_list,over_ratio,experiment = 'seq', base_len = 150):    
    test_fold = {'swallow_id':[],
                 'sig_pad':[],
                 'GT_pad':[],
                 'length':[],
                 'mask':[]}
    for ind in test_list.keys():
        item = test_list[ind]
        
        test_fold['swallow_id'].append(item['singal_file_name']+'_'+str(item['swallow_num']))
        
        pad_signal = item['sig_pad']
        pad_signal_win = sliding_win(pad_signal,over_ratio,base_len=base_len)
        test_fold['sig_pad'].append(pad_signal_win)
        test_fold['GT_pad'].append(item['GT_pad'])
        test_fold['length'].append(item['label_len'])
        test_fold['mask'].append(item['mask'])
    return test_fold

def Ten_fold_patients(recording_list,debug = False):
    
    sample_num = len(recording_list)
    base,remain = sample_num//10,sample_num%10
    
    swallow_num_fold = [base for i in range(10)]
    if remain!=0:
        for i in range(remain):
            swallow_num_fold[i]+=1

        
    # 10-fold patients 
    total_name = [] 
    for ind in recording_list.keys():
        item = recording_list[ind]
        total_name.append(item['patient_id'])
    
    patient_dic = {i:total_name.count(i) for i in total_name}
    patient_names = list(patient_dic.keys())
    
    fold_name_list = [[] for i in range(10)]
    max_patient = max(patient_dic, key=patient_dic.get)
    
    
    
    for i in range(10):
        # initialize
        succeed_flag = 0 
        if debug:
            print('Now fold %d'%i)
        
        
        if i ==0:
            patient_names.remove(max_patient)
        
        while succeed_flag == 0:
            if i == 0:
                fold_name_list[i] = [max_patient]
                can = patient_dic[max_patient]
            else:
                fold_name_list[i] = []
                can = 0
            
            temp_patient_name = deepcopy(patient_names)
            
            
            while can < swallow_num_fold[i]:
                
                candidate = random.choice(temp_patient_name)
                can += patient_dic[candidate]
                fold_name_list[i].append(candidate)
                
                
                temp_patient_name.remove(candidate)
                
            if can == swallow_num_fold[i]:
                
                succeed_flag =1
                patient_names = deepcopy(temp_patient_name)
                if debug:
                    print('fold_%d is OK!'%i)
                
            else:
                if debug:
                    print('fold failed.retry')
                
    return fold_name_list

def ten_fold_small_std(recording_list, std_th = 1.5):
    '''
    split the patients into 10-flod
    Every patient has different number of swallows, so, we need all the folds have same swallow numbers
    '''
    std = 1000
    while std > std_th:
        fold_patient_name = Ten_fold_patients(recording_list)
        sample_len = [len(item) for item in fold_patient_name]
        std = np.std(sample_len)
    return fold_patient_name


def cal_x(ratio,base_len = 150):
    '''
    use to calculate how many extra data length is needed.
    return:
        1. the half length of the padding
        2. the window side
    '''
    
    additional  = base_len*ratio/(1-ratio)
    
    extra = np.int(np.round(additional/2))

    return extra,base_len+extra*2

def sliding_win(temp,over_ratio,base_len = 150):
    
    '''
    view the 1d signal as windows
    base_len is the stride
    ratio is the overlapped ratio
    '''
    # padding zeros
    chennel_number = temp.shape[-1]    
    padding_length, win_size = cal_x(over_ratio,base_len = base_len)
    
    if over_ratio > 0:
        padding_zeros = np.zeros([padding_length,chennel_number], dtype=np.float32)
        
        temp_pad = np.concatenate((padding_zeros,
                             temp,
                             padding_zeros), 0)
    elif over_ratio ==0:
        temp_pad = temp
    else:
        raise ValueError('the overlapped ratio is smaller than zero')
    
    new_array = view_as_windows(temp_pad,(win_size,chennel_number),step=(base_len,chennel_number))
    new_array = np.squeeze(new_array)
    
    return new_array



def get_fold_info(recording_list,over_ratio, experiment = 'seq',base_len = 150):
    
    fold_patient_name = ten_fold_small_std(recording_list)
    if experiment == 'seq':
    
        fold_data = {'fold_%d'%i:{'swallow_id':[],
                                  'sig_pad':[],
                                  'GT_pad':[],
                                  'length':[],
                                  #'signal':[],
                                  #'GT':[],
                                  'mask':[]
                                  } for i in range(10)}
        
        for ind in recording_list.keys():
            temp_item = recording_list[ind]

            
            temp_name = temp_item['patient_id']
            for fold_ind in range(10):
                if temp_name in fold_patient_name[fold_ind]:
                    fold_data['fold_%d'%fold_ind]['swallow_id'].append(temp_item['swallow id'])
                    
                    pad_signal = temp_item['sig_pad']
                    pad_signal_win = sliding_win(pad_signal,over_ratio,base_len=base_len)
                    
                    fold_data['fold_%d'%fold_ind]['sig_pad'].append(pad_signal_win)
                    fold_data['fold_%d'%fold_ind]['GT_pad'].append(temp_item['GT_pad'])
                    fold_data['fold_%d'%fold_ind]['length'].append(temp_item['label_len'])
                    fold_data['fold_%d'%fold_ind]['mask'].append(temp_item['mask'])
        
        fold_data['note'] = experiment
        
    return fold_data

def fetch_fold_data(fold_number,fold_data,all_data = False):
    
    '''
    fold_data is used for dividing the all_data into 10 fold, but not tell us the train and val data
    all_data: 
        False is used for 10-fold validation
        True is used for training the model with all the data and applied on testing
    '''
    
    if not all_data:
        val_fold = 'fold_%d'%fold_number
    
    
        input_train = []
        gt_train = []
        mask_train = []
        
        
        
        for fold_ind in range(10):
            temp_fold = 'fold_%d'%fold_ind
            
            if temp_fold == val_fold: #validation fold
                
                val_fold_data = fold_data[temp_fold]
                input_val = np.array(val_fold_data['sig_pad'])
                gt_val = np.squeeze(np.array(val_fold_data['GT_pad']))
                mask_val = np.squeeze(np.array(val_fold_data['mask']))
            else: # training
                    
                input_train += fold_data[temp_fold]['sig_pad']
                gt_train += fold_data[temp_fold]['GT_pad']
                mask_train += fold_data[temp_fold]['mask']
        
        
        input_train = np.array(input_train)      
        gt_train = np.squeeze(np.array(gt_train))
        mask_train = np.squeeze(np.array(mask_train))
        
        return input_train,input_val,gt_train,gt_val,mask_train,mask_val
    
    else:
        input_train = []
        gt_train = []
        mask_train = []
        
        for fold_ind in range(10):
            temp_fold = 'fold_%d'%fold_ind
            
            input_train += fold_data[temp_fold]['sig_pad']
            gt_train += fold_data[temp_fold]['GT_pad']
            mask_train += fold_data[temp_fold]['mask']
        
        input_train = np.array(input_train)      
        gt_train = np.squeeze(np.array(gt_train))
        mask_train = np.squeeze(np.array(mask_train))
        
        return input_train,gt_train,mask_train

def fetch_test_data(test_fold):
    
    input_test = np.array(test_fold['sig_pad'])
    gt_test = np.squeeze(np.array(test_fold['GT_pad']))
    mask_test = np.squeeze(np.array(test_fold['mask']))
    
    return input_test,gt_test,mask_test

def time_for_saving():
    x = datetime.datetime.now()
    return '%d_%d_%d_%d'%(x.month,x.day,x.hour,x.minute)

def draw_val_curve(train_loss,train_perf,val_perf,save_path,fold_number):
    
    temp_train_perf = np.array(train_perf)
    temp_train_acc = temp_train_perf[:,0]
    
    temp_var_perf = np.array(val_perf)
    temp_val_acc = temp_var_perf[:,0]

    
    plt.plot(train_loss)
    plt.plot(temp_train_acc)
    plt.plot(temp_val_acc)
    
    plt.legend(['train_loss','train_acc','val_acc'])
    plt.xlabel('epoch')
    
    fig_save_path = os.path.join(save_path,'val_fold_%d'%fold_number)
    plt.savefig(fig_save_path)  
    plt.close()

def draw_test_curve(train_loss,train_perf,test_perf,save_path):
    temp_train_perf = np.array(train_perf)
    temp_train_acc = temp_train_perf[:,0]
    
    temp_test_perf = np.array(test_perf)
    temp_test_acc = temp_test_perf[:,0]

    
    plt.plot(train_loss)
    plt.plot(temp_train_acc)
    plt.plot(temp_test_acc)
    
    plt.legend(['train_loss','train_acc','test_acc'])
    plt.xlabel('epoch')
    
    fig_save_path = os.path.join(save_path,'test')
    plt.savefig(fig_save_path)  
    plt.close()