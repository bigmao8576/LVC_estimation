# model building
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
import ops
import numpy as np
import utils
from tensorflow import keras

class My_Model(Model):
    def __init__(self,model_para):
        super(My_Model, self).__init__()
        
        
        self.cnn_ch = model_para['CNN_channel']
        self.cnn_size = model_para['filter_size']
        self.cnn_pool_size = model_para['pooling_size']
        self.cnn_f_size = model_para['filter_size']
        self.rnn_ch = model_para['RNN_channel']
        
        self.s_l = model_para['sample_len']
        self.win_size = model_para['win_size']
        self.s_ch = model_para['signal_ch']
        
        self.fc_ch = model_para['fc_channel']
        
        
        self.conv1 = layers.Conv1D(self.cnn_ch,self.cnn_f_size, strides=1)
        self.conv2 = layers.Conv1D(self.cnn_ch, self.cnn_f_size, strides=1)
        self.pool1= layers.MaxPool1D(pool_size=self.cnn_pool_size)
        self.pool2 = layers.MaxPool1D(pool_size=self.cnn_pool_size)
        
        self.rnn_1_f = layers.GRU(self.rnn_ch, return_sequences=True)
        self.rnn_1_b = layers.GRU(self.rnn_ch, return_sequences=True,go_backwards=True)
        self.bi_1 = layers.Bidirectional(self.rnn_1_f, backward_layer=self.rnn_1_b)
        
        self.rnn_2_f = layers.GRU(self.rnn_ch, return_sequences=True)
        self.rnn_2_b = layers.GRU(self.rnn_ch, return_sequences=True,go_backwards=True)
        self.bi_2 = layers.Bidirectional(self.rnn_2_f, backward_layer=self.rnn_2_b)
        
        self.d1= layers.Dense(self.fc_ch)
        
        self.dropout1 = layers.Dropout(0.5)
        self.d2= layers.Dense(self.fc_ch)
        self.dropout2 = layers.Dropout(0.5)
        self.d3= layers.Dense(1)
        self.at = layers.Attention(use_scale=False)
    
    def call(self, x, m, training = False):
        x = tf.reshape(x, (tf.shape(x)[0]*self.s_l,self.win_size,self.s_ch), name='first_reshape')
        
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x)   
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = tf.nn.leaky_relu(x)
        x = self.pool2(x)
        
        x = tf.reshape(x, (tf.shape(x)[0]//self.s_l,self.s_l,x.shape[1],self.cnn_ch), name='sec_reshape')
        x = tf.reshape(x, (tf.shape(x)[0],self.s_l,x.shape[-2]*self.cnn_ch), name='third_reshape')
        
        state1 = [tf.random.uniform([tf.shape(x)[0],self.cnn_ch],minval=-1, maxval=1),
                  tf.random.uniform([tf.shape(x)[0],self.cnn_ch],minval=-1, maxval=1)]
        state2 = [tf.random.uniform([tf.shape(x)[0],self.cnn_ch],minval=-1, maxval=1),
                  tf.random.uniform([tf.shape(x)[0],self.cnn_ch],minval=-1, maxval=1)]
        
        x = self.bi_1(x,mask=m,initial_state = state1,training=training)
        x = self.bi_2(x,mask=m,initial_state = state2,training=training)
        
        x = self.at([x,x],[m,m])
        
        
        x = self.d1(x)   
        x = tf.nn.leaky_relu(x) 
        x = self.dropout1(x,training=training)       
        
        
        x = self.d2(x)
        x = tf.nn.leaky_relu(x) 
        x = self.dropout2(x,training=training) 
        
        
        x = self.d3(x)
        x = tf.squeeze(x)
        
        return x
    
    


def myloss(target_y, predicted_y,mask,cw):
    l = tf.nn.weighted_cross_entropy_with_logits(target_y, predicted_y, cw, name= 'w_loss')
    mask_l = tf.multiply(l,tf.cast(mask,tf.float32))
    
    return tf.divide(tf.reduce_sum(mask_l),tf.reduce_sum(tf.cast(mask,tf.float32)))

def myloss_seq(model,target_y, predicted_y,mask,cw,l2_c):
    '''
    This loss function is ONLY used for the CRNN or RNN
    '''
    l = tf.nn.weighted_cross_entropy_with_logits(target_y, predicted_y, 1.0, name= 'w_loss')
    
    mask_loss = tf.multiply(l,tf.cast(mask,tf.float32))
    mask_loss = tf.divide(tf.reduce_sum(mask_loss),tf.reduce_sum(tf.cast(mask,tf.float32)))
    
    weight_2_sum = sum(tf.nn.l2_loss(tf_var) for tf_var in model.trainable_variables if not "bias" in tf_var.name)
    return mask_loss+l2_c*weight_2_sum



def train_step(model,sig, labels,mask,c_w,optimizer,l2_c):
    with tf.GradientTape() as tape:

        predictions = model(sig, mask,training = True)
        loss = myloss_seq(model, labels, predictions,mask,c_w,l2_c)
    gradients = tape.gradient(loss, model.trainable_variables)    
    gradients = tf.clip_by_global_norm(gradients, 5, use_norm=None, name=None)[0]
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return myloss_seq(model,labels, predictions,mask,c_w,l2_c), ops.model_perf(predictions,labels,mask)



def eval_step(model,sig, labels,mask,c_w):

    predictions =  model(sig, mask,training = False)
    loss = myloss(labels, predictions,mask,c_w)
    perf = ops.model_perf(predictions, labels, mask)
    
    return loss.numpy(),perf


def train_for_val(fold_number,fold_data,model_para,save_path):
    input_train_np,input_val_np,gt_train_np,gt_val_np,mask_train_np,mask_val_np = utils.fetch_fold_data(fold_number,fold_data)
    
    c_w = 1.0#(np.sum(mask_train_np)-np.sum(gt_train_np))/np.sum(gt_train_np)
    sample_len, win_size, ch = input_train_np.shape[1:]
    
    
    model_para['sample_len'] = sample_len
    model_para['win_size'] = win_size
    model_para['signal_ch'] = ch
    ### setup model parameters
    

    
    batch_size = model_para['batch_size']
    # data pipeline
    
    mask_train_np = np.array(mask_train_np, dtype=bool)
    mask_val_np = np.array(mask_val_np, dtype=bool)

        
    ds_train = tf.data.Dataset.from_tensor_slices((input_train_np,mask_train_np,gt_train_np))
    ds_train = ds_train.shuffle(10000)
    ds_train = ds_train.batch(batch_size)
    
    ds_val = tf.data.Dataset.from_tensor_slices((input_val_np,mask_val_np,gt_val_np))
    ds_val = ds_val.batch(input_val_np.shape[0])
    
    model = My_Model(model_para)

    print(ops.num_weights(model))
    
    
    initial_learning_rate = model_para['lr']
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,
        decay_rate=0.92,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
      
    train_loss = []
    train_perf = []
    val_perf = []
    
    for epoch in range(model_para['total_epoch']):
    
        epoch_loss = []
        epoch_perf = []
        
        
        for input_train,mask_train,gt_train in ds_train:
            
            train_step(model,input_train,gt_train,mask_train,c_w,optimizer,model_para['l2_c'])            
            temp_loss,temp_perf = eval_step(model,input_train,gt_train,mask_train,c_w)    
            epoch_loss.append(temp_loss)    #
            epoch_perf.append(temp_perf)
                
        temp_train_perf = np.mean(np.array(epoch_perf),0)
                
        train_loss.append(np.mean(epoch_loss))
        train_perf.append(temp_train_perf)
            
        for input_val,mask_val,gt_val in ds_val:
            _,temp_val_perf = eval_step(model,input_val,gt_val,mask_val,c_w)
            
            val_perf.append(np.array(temp_val_perf))
                   
        print(epoch,fold_number, model_para['over_ratio'],'-----',np.mean(epoch_loss))
        print('Train----', temp_train_perf)
        print('Valid----', temp_val_perf)
        if (epoch+1)%10 ==0:
            utils.draw_val_curve(train_loss,train_perf,val_perf,save_path,fold_number)
    
    # after the training is finished, get the final val results
    for input_val,mask_val,gt_val in ds_val:
            temp_val_pred = model(input_val,mask_val)
    
    temp_fold_results = {'train_loss': train_loss,
                         'train_perf': train_perf,
                         'val_perf': val_perf,
                         'val_gt': gt_val_np,
                         'val_mask':mask_val_np,
                         'val_pred':temp_val_pred.numpy()                         
                        }
    
    return temp_fold_results

def train_for_test(test_fold_data,fold_data,model_para,save_path):
    input_train_np,gt_train_np,mask_train_np= utils.fetch_fold_data(None,fold_data,all_data = True)   

    
    input_test_np,gt_test_np,mask_test_np = utils.fetch_test_data(test_fold_data)
    
    
    c_w = 1.0#(np.sum(mask_train_np)-np.sum(gt_train_np))/np.sum(gt_train_np)
    sample_len, win_size, ch = input_train_np.shape[1:]
    
    ### setup model parameters
    model_para['sample_len'] = sample_len
    model_para['win_size'] = win_size
    model_para['signal_ch'] = ch
    
    
    batch_size = model_para['batch_size']
    # data pipeline
    
    mask_train_np = np.array(mask_train_np, dtype=bool)
    mask_test_np = np.array(mask_test_np, dtype=bool)
    
    ds_train = tf.data.Dataset.from_tensor_slices((input_train_np,mask_train_np,gt_train_np))
    ds_train = ds_train.shuffle(10000)
    ds_train = ds_train.batch(batch_size)
    
    ds_test = tf.data.Dataset.from_tensor_slices((input_test_np,mask_test_np,gt_test_np))
    ds_test = ds_test.batch(input_test_np.shape[0])
    
    model = My_Model(model_para)
    
    initial_learning_rate = model_para['lr']
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,
        decay_rate=0.92,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    train_loss = []
    train_perf = []
    test_perf = []
    
    
    for epoch in range(model_para['total_epoch']):
    
        epoch_loss = []
        epoch_perf = []
        
        
        for input_train,mask_train,gt_train in ds_train:
            
            train_step(model,input_train,gt_train,mask_train,c_w,optimizer,model_para['l2_c'])            
            temp_loss,temp_perf = eval_step(model,input_train,gt_train,mask_train,c_w)    
            epoch_loss.append(temp_loss)    #
            epoch_perf.append(temp_perf)
                
        temp_train_perf = np.mean(np.array(epoch_perf),0)
                
        train_loss.append(np.mean(epoch_loss))
        train_perf.append(temp_train_perf)
            
        for input_test,mask_test,gt_test in ds_test:
            _,temp_test_perf = eval_step(model,input_test,gt_test,mask_test,c_w)
            
            test_perf.append(np.array(temp_test_perf))
                   
        print(epoch,model_para['over_ratio'],'testing -----',np.mean(epoch_loss))
        print('Train----', temp_train_perf)
        print('Test----', temp_test_perf)
        if (epoch+1)%10 ==0:
            utils.draw_test_curve(train_loss,train_perf,test_perf,save_path)
    
    # after the training is finished, get the final val results
    for input_test,mask_test,gt_test in ds_test:
            temp_test_pred = model(input_test,mask_test)
    
    temp_test_results = {'train_loss': train_loss,
                         'train_perf': train_perf,
                         'test_perf': test_perf,
                         'test_gt': gt_test_np,
                         'test_mask':mask_test_np,
                         'test_pred':temp_test_pred.numpy()                         
                        }
    
    return temp_test_results