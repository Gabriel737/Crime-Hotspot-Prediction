import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import config
import data_loader

# def classification_loss(pred, target):
#     '''
#     Inputs : pred <np.array> : predicted values
#              target <np.array> : target values
#     '''
#     for i in range(pred.shape[-1]):
#         loss = nn.BCELoss()
#         print(pred[:,1].shape)
#         print(target[:,1].shape)
#         if i == 0:
#             bce_loss = loss(pred[:,i].view(-1,1),target[:,i].view(-1,1))
#         else:
#             bce_loss = bce_loss + loss(pred[:,i].view(-1,1),target[:,i].view(-1,1))
#     return bce_loss

def classification_loss(pred,target,weights=[0.5,1]):
    input = torch.clamp(pred,min=1e-7,max=1-1e-7)
    bce = - weights[1] * target * torch.log(pred) - (1 - target) * weights[0] * torch.log(1 - pred)
    return torch.mean(bce)

# def regression_loss(pred,target):
#     '''
#     Inputs : pred <np.array> : predicted values
#              target <np.array> : target values
#     '''
#     for i in range(pred.shape[-1]):
#         loss = nn.MSELoss()
#         if i == 0:
#             mse_loss = loss(pred,target)
#         else:
#             mse_loss = mse_loss + loss(pred,target)
#     return mse_loss

# def total_loss(bce_loss,mse_loss, task_num):
#     '''
#     Inputs : bce_loss <type> : bce loss
#              mse_loss <type> : mse loss
#     '''
#     log_vars = nn.parameter.Parameter(torch.zeros((task_num)))

#     precision_bce = torch.exp(-log_vars[0])
#     bce_loss = precision_bce*bce_loss + log_vars[0]

#     precision_mse = torch.exp(-log_vars[1])
#     mse_loss = precision_mse*mse_loss + log_vars[1]

#     return bce_loss + mse_loss
