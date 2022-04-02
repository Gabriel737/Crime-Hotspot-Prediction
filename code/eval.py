import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader

import config
from data_loader import FeatureDataset
from model import CNNLSTM
import utils

def load_model(model_path, model):
    trained = torch.load(model_path)
    model.load_state_dict(trained['model'])
    print('\n Model Loaded \n')

def get_confusion_matrix(pred_scores, y_bin, threshold):

    pred_bin = (pred_scores > threshold).float()

    tp = 0
    fp = 0
    fn_neigh_pos = 0
    fn = 0
    tn = 0

    for i in range(y_bin.shape[0]):
        for j in range(y_bin.shape[1]):
            for k in range(y_bin.shape[2]):

                if pred_bin[i][j][k] == 1 and y_bin[i][j][k] == 1:
                    tp += 1
                if pred_bin[i][j][k] == 1 and y_bin[i][j][k] == 0:
                    fp += 1
                if pred_bin[i][j][k] == 0 and y_bin[i][j][k] == 0:
                    tn += 1
                if(pred_bin[i][j][k]==0 and y_bin[i][j][k]==1):
                    n1=pred_bin[i][j-1][k] if (j-1)>=0 else 0
                    n2=pred_bin[i][j][k-1] if (k-1)>=0 else 0
                    n3=pred_bin[i][j+1][k] if (j+1)<pred_bin.shape[1] else 0
                    n4=pred_bin[i][j][k+1] if (k+1)<pred_bin.shape[2] else 0
                    n5=pred_bin[i][j-1][k-1] if (j-1)>=0 and (k-1)>=0  else 0
                    n6=pred_bin[i][j+1][k-1] if (j+1)<pred_bin.shape[1] and (k-1)>=0  else 0
                    n7=pred_bin[i][j-1][k+1] if (j-1)>=0 and (k+1)<pred_bin.shape[2]  else 0
                    n8=pred_bin[i][j+1][k+1] if (j+1)<pred_bin.shape[1] and (k+1)<pred_bin.shape[2]  else 0
                    if(n1+n2+n3+n4+n5+n6+n7+n8>=1):
                        fn_neigh_pos += 1
                    else:
                        fn += 1
    
    return tp, fp, fn, tn, fn_neigh_pos


def evaluate (model,val_loader,batch_size,thresholds,get_percent=True):

    model.eval()

    total = 0
    total_tp = 0
    total_fp = 0
    total_fn_neigh_pos = 0
    total_fn = 0
    total_tn = 0

    tp_list = list()
    fp_list = list()
    fn_neigh_pos_list = list()
    fn_list = list()
    tn_list = list()
    accuracy_list = list()

    for thresh in tqdm(thresholds):
        for X, y_reg, y_bin in tqdm(val_loader):
            if y_bin.shape[0] == batch_size:

                pred_scores, _ = model(X)
                # pred_bin_np = pred_bin.view(-1,1).detach().cpu().numpy()
                # y_bin_np = y_bin.view(-1,1).detach().cpu().numpy()
                # print(confusion_matrix(y_true=y_bin_np, y_pred=pred_bin_np))
                pred_scores = pred_scores.view(batch_size, config.CELL_COUNT, config.CELL_COUNT)
                y_bin = y_bin.view(batch_size, config.CELL_COUNT, config.CELL_COUNT)
                tp, fp, fn, tn, fn_neigh_pos = get_confusion_matrix(pred_scores,y_bin,thresh)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn
                total_fn_neigh_pos += fn_neigh_pos
                total += batch_size*config.CELL_COUNT*config.CELL_COUNT
        
        if get_percent == True:
            summation = total_tp + total_fn + total_fn_neigh_pos
            total_tp = (total_tp/summation) * 100
            total_fn = (total_fn/summation) * 100
            total_fn_neigh_pos = (total_fn_neigh_pos/summation) * 100

        accuracy_list.append((total_tp+total_tn)/total)
        tp_list.append(total_tp)
        fp_list.append(total_fp)
        fn_list.append(total_fn)
        tn_list.append(total_tn)
        fn_neigh_pos_list.append(total_fn_neigh_pos)
    return tp_list, fp_list, fn_list, tn_list, fn_neigh_pos_list, accuracy_list

if __name__ == '__main__':

    start_time = time.time()

    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)

    optim_name = config.OPTIM_NAME
    model_path = config.MODEL_SAVE_PATH + f'/model_checkpoint_{optim_name}_{config.LR}_{config.TRAIN_BATCH_SIZE}_{config.CLASS_THRESH}_{config.RANDOM_SEED}.pt'

    val_data = FeatureDataset(feat_data_path=config.VAN_DATA_PRCD+'/features.h5',
                              target_data_path=config.VAN_DATA_PRCD+'/targets.h5',
                              device=device,
                              name='val') 
    
    val_loader = DataLoader (val_data, batch_size=config.TRAIN_BATCH_SIZE)

    model = CNNLSTM(n_input_channels=len(config.CRIME_CATS), embed_size=2304, batch_size=config.TRAIN_BATCH_SIZE, device=device)
    model.to(device)

    load_model(model_path, model)
    tp_list, fp_list, fn_list, tn_list, fn_neigh_pos_list, accuracy_list = evaluate(model=model, val_loader=val_loader, 
                                                                                    batch_size=config.TRAIN_BATCH_SIZE, 
                                                                                    thresholds=config.EVAL_THRESHOLDS,
                                                                                    get_percent=True)
    
    print(tp_list)
    print(fp_list)
    print(fn_list)
    print(tn_list)
    print(fn_neigh_pos_list)
    print(accuracy_list)

    utils.plotConfusionMatrix(threshold_list=config.EVAL_THRESHOLDS, tp_list=tp_list,
                              fn_list=fn_list, fn_neigh_pos_list=fn_neigh_pos_list,
                              accuracy_list=accuracy_list)