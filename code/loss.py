import numpy as np
import pandas as pd

import torch

import config

def classification_loss(pred,target,weights=config.CROSS_ENT_LOSS_WEIGHTS):
    '''
    Calculate weighted cross entropy loss
    
    Inputs - pred: predicted probability scores
             target: binned target values
             weights: cross entropy weights
    '''
    pred = torch.clamp(pred,min=1e-7,max=1-1e-7)
    bce = - weights[1] * target * torch.log(pred) - (1 - target) * weights[0] * torch.log(1 - pred)
    return torch.mean(bce)