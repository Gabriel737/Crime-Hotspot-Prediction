import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import config
import data_loader

def classification_loss(pred, target):
    '''
    Inputs : pred <np.array> : predicted values
             target <np.array> : target values
    '''
    for i in range(pred.shape[-1]):
        loss = nn.BCELoss()
        if i == 0:
            bce_loss = loss(pred,target.float())
        else:
            bce_loss = bce_loss + loss(pred,target.float())
    return bce_loss

def regression_loss(pred,target):
    '''
    Inputs : pred <np.array> : predicted values
             target <np.array> : target values
    '''
    for i in range(pred.shape[-1]):
        loss = nn.MSELoss()
        if i == 0:
            mse_loss = loss(pred,target)
        else:
            mse_loss = mse_loss + loss(pred,target)
    return mse_loss

def total_loss(bce_loss,mse_loss, task_num):
    '''
    Inputs : bce_loss <type> : bce loss
             mse_loss <type> : mse loss
    '''
    log_vars = nn.parameter.Parameter(torch.zeros((task_num)))

    precision_bce = torch.exp(-log_vars[0])
    bce_loss = precision_bce*bce_loss + log_vars[0]

    precision_mse = torch.exp(-log_vars[1])
    mse_loss = precision_mse*mse_loss + log_vars[1]

    return bce_loss + mse_loss
