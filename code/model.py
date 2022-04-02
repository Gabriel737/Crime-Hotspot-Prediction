import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from pytorch_model_summary import summary


import config


class CNNLSTM(nn.Module):
    '''
    Implementation of VGG-like CNN network followed by LSTM layers 
    followed by two parallel FC layers to output likelihood of cell hotspot
    and number of crimes in a cell
    '''

    def __init__(self, n_input_channels, embed_size, batch_size, device):
        '''
        Intialise class variables

        Inputs : n_inpit_channels <int> : number of channels in input heatmaps
                 embed_size <int> : LSTM input size 
                 batch_size <int> : input batch size
                 output_size <int> : number of predictions in output
        '''
        super(CNNLSTM, self).__init__()
        # CNN
        self.conv1_1 = nn.Conv2d(in_channels=n_input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=config.drop_p)
        self.batch_norm12 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)

        (self.h1, self.c1) =  (torch.zeros(1, batch_size, 800, device=device).float(), torch.zeros(1, batch_size, 800, device=device).float())
        (self.h2, self.c2) =  (torch.zeros(1, batch_size, 800, device=device).float(), torch.zeros(1, batch_size, 800, device=device).float())
        (self.h3, self.c3) =  (torch.zeros(1, batch_size, 676, device=device).float(), torch.zeros(1, batch_size, 676, device=device).float())

        self.lstm1 = nn.LSTM(input_size=embed_size, hidden_size=800, 
                              num_layers=1)
        self.lstm2 = nn.LSTM(input_size=800, hidden_size=800, 
                              num_layers=1)
        self.lstm3 = nn.LSTM(input_size=800, hidden_size=676,
                             num_layers=1)

        self.fc1 = nn.Linear(in_features=676,out_features=676)
        self.fc2 = nn.Linear(in_features=676,out_features=676)

        nn.init.xavier_normal_(self.conv1_1.weight)
        nn.init.xavier_normal_(self.conv1_2.weight)
        nn.init.xavier_normal_(self.conv2_1.weight)
        nn.init.xavier_normal_(self.conv2_2.weight)
        nn.init.xavier_normal_(self.conv3_1.weight)
        nn.init.xavier_normal_(self.conv3_2.weight)
        nn.init.xavier_normal_(self.conv4_1.weight)
        nn.init.xavier_normal_(self.conv4_2.weight)
        nn.init.xavier_normal_(self.conv5.weight)
        
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        
        for name, param in self.lstm2.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        
        for name, param in self.lstm3.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
    
    def forward(self, x):
        x = x.view(16*32,
                   6,
                   26,26).float()
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.batch_norm12(x)
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.batch_norm12(x)
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.batch_norm3(x)
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv5(x))
        x = x.view(16,32,-1)
        x,(h1,_) = self.lstm1(x, (self.h1, self.c1))
        x,(h2,_) = self.lstm2(x, (self.h2, self.c2))
        x,(h3, _) = self.lstm3(x, (self.h3, self.c3))
        x = x.squeeze()[-1,:,:]
        x1 = self.sigmoid(self.fc1(x))
        x2 = self.sigmoid(self.fc2(x))
        return x1, x2
    
if __name__ == '__main__':

    model = CNNLSTM(n_input_channels=6, 
                    batch_size=32, 
                    embed_size=2304)
    print(summary(model, 
                  torch.zeros(32,
                              16,
                              6,
                              26,
                              26), 
                  show_input=True, show_hierarchical=True))