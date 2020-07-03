#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:57:40 2020

@author: bigmao
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def tf_sigmoid(x):
    return 1/(1 + tf.exp(-x)) 

def model_perf(model_output, ground_truth, mask):
    pre_value = tf_sigmoid(model_output)
    pre_logits = tf.round(pre_value)
    
    corrected_unmask = tf.cast(ground_truth, 'int32') == tf.cast(pre_logits, 'int32') 
    
    tp_unmask = tf.math.logical_and(corrected_unmask,(tf.cast(pre_logits, 'int32') == tf.constant(1,'int32')))
    tn_unmask = tf.math.logical_and(corrected_unmask,(tf.cast(pre_logits, 'int32') == tf.constant(0,'int32')))
    
    tp_mask = tf.math.logical_and(tp_unmask,mask)
    tn_mask = tf.math.logical_and(tn_unmask,mask)
    
    uncorrected_unmask = tf.cast(ground_truth, 'int32') != tf.cast(pre_logits, 'int32') 
    
    fp_unmask = tf.math.logical_and(uncorrected_unmask,(tf.cast(pre_logits, 'int32') == tf.constant(1,'int32')))
    fn_unmask = tf.math.logical_and(uncorrected_unmask,(tf.cast(pre_logits, 'int32') == tf.constant(0,'int32')))
    
    fp_mask = tf.math.logical_and(fp_unmask,mask)
    fn_mask = tf.math.logical_and(fn_unmask,mask)
    
    tp = tf.reduce_sum(tf.cast(tp_mask,'int32'))
    tn = tf.reduce_sum(tf.cast(tn_mask,'int32'))
    fp = tf.reduce_sum(tf.cast(fp_mask,'int32'))
    fn = tf.reduce_sum(tf.cast(fn_mask,'int32'))
    
    acc = (tp+tn)/(tp+tn+fp+fn)
    recall = tp/(tp+fn)
    specificity = tn/(tn+fp)
    f1 = 2*tp/(2*tp+fp+fn)
    
    return [acc.numpy(), recall.numpy(), specificity.numpy(),f1.numpy()]

def per2value(tensor_list):
    value_list = [item.numpy() for item in tensor_list]
    return value_list

def num_weights(model):
    w_ls = model.get_weights()
    total_num = 0
    for item in w_ls:
        total_num += np.product(item.shape)
        
    print(total_num)

