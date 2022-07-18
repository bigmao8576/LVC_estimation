from sklearn import metrics
import numpy as np
import ops
import os

def roc_auc_seq(label,pred,mask):
    
    if np.max(pred)>1 or np.min(pred)<0:
        raise ValueError('the predicted value are not in the range [0,1]')
    
    
    series_gt = []
    series_pred = []
    
    for temp_output,temp_gt,temp_mask in zip(pred.ravel(),label.ravel(),mask.ravel()):
        if temp_mask:
            series_pred.append(temp_output)
            series_gt.append(temp_gt)
            
    #ROC now
    fpr,tpr,_ =metrics.roc_curve(series_gt, series_pred)
    auc = metrics.auc(fpr,tpr)
    
    return fpr, tpr,auc

def edge_error_p(edge_dif,total_num,smaller_than=4,norm = True):
    count = 0
    
    for item in edge_dif:
        if abs(item)<smaller_than:
            count+=1
    if norm:
        return count/total_num
    else:
        return count
def frame_error(pred_dig,gt_post):
    '''
    must be [trial, length]
    '''
    
    missing = 0
    multi_edge = 0
    onset_dif = []
    offset_dif = []
    
    duration_ratio = []
    
    for pred_sw,label_sw in zip(pred_dig,gt_post):
        edge_pred = np.abs(np.diff(pred_sw))
        edge_label = np.abs(np.diff(label_sw))
        
        edge_pred = np.int32(edge_pred)
        edge_label = np.int32(edge_label)
        
        if np.sum(edge_pred) == 2:
            pred_on, pred_off = np.where(edge_pred==1)[0]
            label_on, label_off = np.where(edge_label==1)[0]
            
            onset_dif.append(pred_on-label_on)
            offset_dif.append(pred_off-label_off)
            
            duration_ratio.append(np.sum(pred_sw)/np.sum(label_sw))
            
        elif np.sum(edge_pred) == 0:
            missing += 1
        else:
            multi_edge += 1
            pred_on, pred_off = np.where(edge_pred==1)[0][0],np.where(edge_pred==1)[0][-1]
            label_on, label_off = np.where(edge_label==1)[0]
        
            onset_dif.append(pred_on-label_on)
            offset_dif.append(pred_off-label_off)   
            duration_ratio.append(np.sum(pred_sw)/np.sum(label_sw))
            
    return onset_dif,offset_dif,missing,multi_edge,[np.mean(duration_ratio),np.std(duration_ratio)]


# =============================================================================
# missing = 0
# multi_edge = 0
# onset_dif = []
# offset_dif = []
# 
# duration_ratio = []
# 
# 
# pred_dig = val_pred_dig
# gt_post = val_gt_post
# 
# for pred_sw,label_sw in zip(pred_dig,gt_post):
#     edge_pred = np.abs(np.diff(pred_sw))
#     edge_label = np.abs(np.diff(label_sw))
#     
#     edge_pred = np.int32(edge_pred)
#     edge_label = np.int32(edge_label)
# 
#     if np.sum(edge_pred) == 2:
#         pred_on, pred_off = np.where(edge_pred==1)[0]
#         label_on, label_off = np.where(edge_label==1)[0]
# 
# 
# 
# 
# =============================================================================










    
def val_perf_seq(fold_results):
    '''
    This function is used for the sequence results ONLY
    This function is used for the validation set ONLY
    '''
        #acc,spe,recall,ROC, edge_number, 3-frame,

    
    val_perf_total = []
    val_output_post = []
    val_gt_post = []
    val_mask_post = []
       
    for key in fold_results.keys():
        item = fold_results[key]
        val_perf_total.append(item['val_perf'][-1])
        
        val_output_post.append(item['val_pred'])
        val_gt_post.append(item['val_gt'])
        val_mask_post.append(item['val_mask'])
        
    val_perf_total = np.array(val_perf_total)
    val_perf_mean = np.mean(val_perf_total,0)
    val_perf_std = np.std(val_perf_total,0)
    
    val_output_post = np.concatenate(val_output_post)
    val_output_post = ops.sigmoid(val_output_post)
    val_gt_post = np.concatenate(val_gt_post)
    val_mask_post = np.concatenate(val_mask_post)
    
    
            
    #ROC now
    fpr,tpr,auc = roc_auc_seq(val_gt_post,val_output_post,val_mask_post)
    
    val_pred_dig = np.round(val_output_post)
    val_mask_post = np.int64(val_mask_post)
    
    val_pred_dig = val_pred_dig*val_mask_post
    
    onset_dif,offset_dif,missing,multi_edge,d_r = frame_error(val_pred_dig,val_gt_post)
    
    onset_p = edge_error_p(onset_dif,len(val_gt_post))
    offset_p = edge_error_p(offset_dif,len(val_gt_post))
    
    val_results = {'acc':[val_perf_mean[0],val_perf_std[0]],
                   'recall':[val_perf_mean[1],val_perf_std[1]],
                   'spe':[val_perf_mean[2],val_perf_std[2]],
                   'auc':auc,
                   'fpr':fpr,
                   'tpr':tpr,
                   'onset_dif':onset_dif,
                   'offset_dif':offset_dif,
                   'onset_p':onset_p,
                   'offset_p':offset_p,
                   'missing_sw':missing/len(val_gt_post),
                   'multi_edges':multi_edge/len(val_gt_post),
                   'duration_ratio': d_r
                   }
    return val_results


def test_perf_seq(test_results):
    '''
    This function is used for the sequence results ONLY
    This function is used for the testing set ONLY
    '''
        #acc,spe,recall,ROC, edge_number, 3-frame,

    
    test_perf = test_results['test_perf'][-1]
    test_output_post = test_results['test_pred']
    test_output_post = ops.sigmoid(test_output_post)
    
    
    test_gt_post = test_results['test_gt']
    test_mask_post = test_results['test_mask']
       

    #ROC now
    fpr,tpr,auc = roc_auc_seq(test_gt_post,test_output_post,test_mask_post)
    
    test_pred_dig = np.round(test_output_post)
    
    test_mask_post = np.int64(test_mask_post)
    test_pred_dig = test_pred_dig*test_mask_post
    
    onset_dif,offset_dif,missing,multi_edge,d_r = frame_error(test_pred_dig,test_gt_post)   
    
    onset_p = edge_error_p(onset_dif,len(test_gt_post))
    offset_p = edge_error_p(offset_dif,len(test_gt_post))
    
    val_results = {'acc':test_perf[0],
                   'recall':test_perf[1],
                   'spe':test_perf[2],
                   'auc':auc,
                   'fpr':fpr,
                   'tpr':tpr,
                   'onset_dif':onset_dif,
                   'offset_dif':offset_dif,
                   'onset_p':onset_p,
                   'offset_p':offset_p,
                   'missing_sw':missing/len(test_gt_post),
                   'multi_edges':multi_edge/len(test_gt_post),
                   'duration_ratio':d_r
                   }
    return val_results

def write_results(total_results,save_path,test_mode):

    info_str = 'Model_setting \n\n'

    model_para = total_results['model_para']
    for key in model_para.keys():
        temp_str = key + ' = '+ str(model_para[key])+'\n'
        
        info_str += temp_str
    

    
    info_str += '\n'
    info_str += 'Validation results \n'
    results = total_results['val_summary']
     
    info_str += 'acc'+ ' = ' + str(results['acc'])+'\n'
    info_str += 'recall'+ ' = ' + str(results['recall'])+'\n'
    info_str += 'specificity' + ' = ' + str(results['spe'])+'\n'
    info_str += 'auc' + ' = ' + str(results['auc'])+'\n\n'
    info_str += 'duration_ratio' + ' = ' + str(results['duration_ratio'])+'\n'
    info_str += 'onset_p' + ' = ' + str(results['onset_p'])+'\n'
    info_str += 'offset_p' + ' = ' + str(results['offset_p'])+'\n'
    info_str += 'missing_rate' + ' = ' + str(results['missing_sw'])+'\n'
    info_str += 'mulitple_rate' + ' = ' + str(results['multi_edges'])+'\n'
    
    if test_mode:
        info_str += '\n'
        info_str += 'Testing results \n'
        results = total_results['test_summary']
         
        info_str += 'acc'+ ' = ' + str(results['acc'])+'\n'
        info_str += 'recall'+ ' = ' + str(results['recall'])+'\n'
        info_str += 'specificity' + ' = ' + str(results['spe'])+'\n'
        info_str += 'auc' + ' = ' + str(results['auc'])+'\n\n'
        info_str += 'duration_ratio' + ' = ' + str(results['duration_ratio'])+'\n'
        info_str += 'onset_p' + ' = ' + str(results['onset_p'])+'\n'
        info_str += 'offset_p' + ' = ' + str(results['offset_p'])+'\n'
        info_str += 'missing_rate' + ' = ' + str(results['missing_sw'])+'\n'
        info_str += 'mulitple_rate' + ' = ' + str(results['multi_edges'])+'\n'
    
    
    txt_path = os.path.join(save_path,'results.txt')
    
    f = open(txt_path, "w")
    f.write(info_str)
    f.close()
