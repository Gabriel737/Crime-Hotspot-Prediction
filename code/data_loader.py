import numpy as np
import pandas as pd
import h5py

import torch
from torch.utils.data import DataLoader, Dataset

import config

class FeatureDataset(Dataset):

    def __init__(self, crime_feat_data_path, sec_feat_data_path, target_data_path, device, name):
        '''
        Initialise class variables

        Inputs: feat_data_path <str> : path to feature data
                target_data_path <str> : path to target data
                device <torch.device> : gpu/cpu
                name <str> : train/test
        '''
        self.device = device
        self.crime_features = self.read_h5(data_path = crime_feat_data_path, name = name)
        self.sec_features = self.read_h5(data_path = sec_feat_data_path, name = name)
        self.targets = self.read_h5(data_path = target_data_path, name = name)

        self.targets = self.reshape_targets(self.targets)
        self.bin_targets = self.get_binary_targets(self.targets)
        self.X_crime, self.X_sec, self.y_bin = self.numpy2tensor(self.crime_features, self.sec_features, self.bin_targets)
        self.n_samples = self.targets.shape[0]

    def read_h5(self, data_path, name):
        '''
        Read a h5 file

        Inputs: data_path <str> : path to data
                name <str> : dataset name in h5 file
        '''
        hf = h5py.File(data_path, 'r')
        arr = np.array(hf[name][:])
        return arr

    def numpy2tensor(self, *args):
        '''
        Convert numpy arrays to tensors

        Input(s) : *args <np.array> : variable number of numpy arrays
        '''
        tensor_list = list()
        for a in args:
            tensor_list.append(torch.from_numpy(a).to(self.device))
        return tensor_list
    
    def reshape_targets(self, targets):
        '''
        Change shape of target array

        Input - targets <no.array> : targets array to be reshaped
        '''
        n_bins = int(config.BB_DIST/config.BB_CELL_LEN)
        total_bins = int(n_bins*n_bins)
        return  targets.reshape(-1,total_bins)

    def get_binary_targets(self, targets):
        '''
        Convert targets array to binary values

        Input : targets <np.array> : targets array to be binarised
        '''
        flat = targets.reshape(-1,1)
        bin_flat = (flat>0).astype(int)
        n_bins = int(config.BB_DIST/config.BB_CELL_LEN)
        total_bins = int(n_bins*n_bins)
        bin_targets = bin_flat.reshape(-1,total_bins)
        return bin_targets

    def __len__(self):
        '''
        Get number of samples in dataset
        '''
        return self.n_samples

    def __getitem__(self, idx):
        '''
        Get batched inputs and binned targets
        '''
        return self.X_crime[idx].float(), self.X_sec[idx].float(), self.y_bin[idx].float()
    
if __name__ == "__main__":

    name = 'train'
    crime_feat_data_path = config.VAN_DATA_PRCD + '/features.h5'
    sec_feat_data_path = config.VAN_DATA_PRCD + '/sec_features.h5'
    target_data_path = config.VAN_DATA_PRCD + '/targets.h5'
    

    dataset = FeatureDataset(crime_feat_data_path=crime_feat_data_path,
                            sec_feat_data_path=sec_feat_data_path,
                            target_data_path=target_data_path, 
                            device = torch.device(config.DEVICE),
                            name=name
                            )
    
    data_loader = DataLoader(dataset=dataset, batch_size=config.TRAIN_BATCH_SIZE)

    x_crime, x_sec, y_bin = next(iter(data_loader))

    print(x_crime.shape, x_sec.shape, y_bin.shape)
    print(len(dataset))



