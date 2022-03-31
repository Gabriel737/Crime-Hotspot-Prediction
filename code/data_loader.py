import numpy as np
import pandas as pd
import h5py

import torch
from torch.utils.data import DataLoader, Dataset

import config

class FeatureDataset(Dataset):

    def __init__(self, feat_data_path, target_data_path, device, name):
        '''
        Initialise class variables

        Inputs: feat_data_path <str> : path to feature data
                target_data_path <str> : path to target data
                device <torch.device> : gpu/cpu
                name <str> : train/test
        '''
        self.device = device
        self.features = self.read_h5(data_path = feat_data_path, name = name)
        self.targets = self.read_h5(data_path = target_data_path, name = name)
        self.targets = self.reshape_targets(self.targets)
        self.bin_targets = self.get_binary_targets(self.targets)
        self.tfm_targets = self.transform_targets(self.targets)
        self.X, self.y, self.y_bin = self.numpy2tensor(self.features, self.tfm_targets, self.bin_targets)
        self.n_samples = self.targets.shape[0]

    def read_h5(self, data_path, name):
        '''
        Read a h5 file

        Inputs: data_path <str> : path to data
                name <str> : dataset name in h5 file
        Output: arr <np.array> : numpy array
        '''
        hf = h5py.File(data_path, 'r')
        arr = np.array(hf[name][:])
        return arr

    def numpy2tensor(self, *args):
        '''
        Convert numpy arrays to tensors

        Input(s) : *args <np.array> : variable number of numpy arrays
        Output : tensor_list <list(torch)> : list of converted tensors
        '''
        tensor_list = list()
        for a in args:
            tensor_list.append(torch.from_numpy(a).to(self.device))
        return tensor_list
    
    def reshape_targets(self, targets):
        n_bins = int(config.BB_DIST/config.BB_CELL_LEN)
        total_bins = int(n_bins*n_bins)
        return  targets.reshape(-1,total_bins)

    def get_binary_targets(self, targets):
        '''
        Convert targets array to binary values

        Input : targets <np.array> : targets array
        Output : bin_targets <np.array> : binary targets array
        '''
        flat = targets.reshape(-1,1)
        bin_flat = (flat>0).astype(int)
        n_bins = int(config.BB_DIST/config.BB_CELL_LEN)
        total_bins = int(n_bins*n_bins)
        bin_targets = bin_flat.reshape(-1,total_bins)
        return bin_targets

    def transform_targets(self, targets):
        '''
        Restrict continuous targets in 0-1 range 
        '''
        targets_log = np.where(targets>0, np.log(targets), 0)
        max_log = np.max(targets_log)
        return np.where(targets_log > 0, targets_log/max_log, 0)

    def __len__(self):
        '''
        Output : n_samples <int> : total samples in dataset
        '''
        return self.n_samples

    def __getitem__(self, idx):
        '''
        Input : idx <int> : data at index idx
        Output : X[idx], y[idx] <torch, torch> : feature and target tensors at index idx
        '''
        return self.X[idx].float(), self.y[idx].float(), self.y_bin[idx].float()
    
if __name__ == "__main__":
    name = 'train'
    feat_data_path = config.VAN_DATA_PRCD + '/features.h5'
    target_data_path = config.VAN_DATA_PRCD + '/targets.h5'
    dataset = FeatureDataset(feat_data_path=feat_data_path, 
                            target_data_path=target_data_path, 
                            device = torch.device(config.DEVICE),
                            name=name
                            )
    
    data_loader = DataLoader(dataset=dataset, batch_size=config.TRAIN_BATCH_SIZE)

    x, y, y_bin = next(iter(data_loader))

    print(x.shape, y.shape, y_bin.shape)



