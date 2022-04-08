from turtle import hideturtle
import numpy as np
import config

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from pytorch_model_summary import summary

import config

class ConvLSTMCell(nn.Module):
    '''
    Class for Convolution LSTM Cell
    '''

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        '''
        Inputs - input_dim <int> : number of input dimensions (channels)
                 hidden_dim <int> : number of hidden dimensions (channels)
                 kernel_size <int> : convolution kernel size
                 bias <bool> : add bias or not
        '''
        super(ConvLSTMCell,self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = int((kernel_size - 1)/2)

        self.conv = nn.Conv2d(in_channels=self.input_dim+self.hidden_dim,
                              out_channels=4*self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
    def forward(self, input, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f*c_cur + i*g
        h_next = o*torch.tanh(c_next)

        return h_next, c_next
    
    def init_hidden(self, batch_size, img_size):
        h, w = img_size
        return (torch.zeros(batch_size, self.hidden_dim, h, w, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, h, w, device=self.conv.weight.device))
    

class ConvLSTM(nn.Module):
    '''
    Class for Convoltion LSTM layer
    '''

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        '''
        Input : input_dim <int> : number of input dimensions (channels)
                hidden_dim <int> : number of hidden dimensions (channels)
                kernel_size <int> : convolution kernel size
                bias <bool> : add bias or not
        '''
        super(ConvLSTM, self).__init__()

        self.cell = ConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)

    def forward(self, input, hidden_state=None):

        b, seq_len, _, h, w = input.size()

        hidden_state = self._init_hidden(batch_size=b, img_size=(h,w))
        h,c = hidden_state
        output_inner = list()
        for t in range(seq_len):
            h,c = self.cell(input=input[:,t,:,:,:], cur_state=[h,c])
            output_inner.append(h)
        output_inner = torch.stack((output_inner),dim=1)
        layer_output = output_inner
        last_state = [h,c]
        return layer_output
    
    def _init_hidden(self, batch_size, img_size):
        init_states = self.cell.init_hidden(batch_size=batch_size,img_size=img_size)
        return init_states

class HotspotPredictor(nn.Module):
    '''
    Class for Crime Hostspot Prediction model
    '''
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        '''
        Inputs - input_dim <int> : number of input dimensions (channels)
                 hidden_dim <int> : number of hidden dimensions (channels)
                 kernel_size <int> : convolution kernel size
                 bias <bool> : add bias or not
        '''
        super(HotspotPredictor, self).__init__()

        self.convlstm1 = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)
        self.batch_norm1 = nn.BatchNorm3d(num_features=hidden_dim)

        self.convlstm2 = ConvLSTM(input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)
        self.batch_norm2 = nn.BatchNorm3d(num_features=hidden_dim)

        self.convlstm3 = ConvLSTM(input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)
        self.batch_norm3 = nn.BatchNorm3d(num_features=hidden_dim)

        self.convlstm4 = ConvLSTM(input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)
        self.batch_norm4 = nn.BatchNorm3d(num_features=hidden_dim)

        self.maxpool = nn.MaxPool3d(kernel_size=1,stride=[2,1,1])
        self.dropout = nn.Dropout(p=config.DROP_P)
        self.sigmoid = nn.Sigmoid()

        self.conv3d = nn.Conv3d(in_channels=hidden_dim,out_channels=3,kernel_size=(1,3,3),padding=(0,1,1),bias=True)

        fc_input = int(3 * config.CELL_COUNT * config.CELL_COUNT) + config.N_SEC_FEATS
        fc_out = int(config.CELL_COUNT * config.CELL_COUNT)
        self.fc = nn.Linear(in_features=fc_input, out_features=fc_out)
    
    def forward(self, input_crime, input_sec, hidden_state=None):
        out = self.convlstm1(input_crime)
        out = out.permute(0,2,1,3,4)
        out = self.batch_norm1(out)
        out = self.maxpool(out)
        out = self.dropout(out)

        out = out.permute(0,2,1,3,4)
        out = self.convlstm2(out)
        out = out.permute(0,2,1,3,4)
        out = self.batch_norm2(out)
        out = self.maxpool(out)
        out = self.dropout(out)

        out = out.permute(0,2,1,3,4)
        out = self.convlstm3(out)
        out = out.permute(0,2,1,3,4)
        out = self.batch_norm3(out)
        out = self.maxpool(out)
        out = self.dropout(out)

        out = out.permute(0,2,1,3,4)
        out = self.convlstm4(out)
        out = out.permute(0,2,1,3,4)
        out = self.batch_norm4(out)
        out = self.maxpool(out)
        out = self.dropout(out)

        out = self.conv3d(out)
        out = out.view(config.TRAIN_BATCH_SIZE,1,1,-1)
        input_sec = input_sec.view(config.TRAIN_BATCH_SIZE,1,1,-1)
        out = torch.cat((out,input_sec),axis=-1)
        out = self.fc(out)
        out = self.sigmoid(out)
        out = out.view(config.TRAIN_BATCH_SIZE,1,1,config.CELL_COUNT,config.CELL_COUNT)
        # print(f'Output shape : {out.shape}')
        return out


if __name__ == '__main__':

    model = HotspotPredictor(input_dim=len(config.CRIME_CATS), hidden_dim=64, kernel_size=config.KERNEL_SIZE, bias=True)
    print(model)

    dummy_crime_input = torch.zeros(config.TRAIN_BATCH_SIZE, config.SEQ_LEN, len(config.CRIME_CATS), config.CELL_COUNT, config.CELL_COUNT)
    dummy_crime_sec = torch.zeros(config.TRAIN_BATCH_SIZE, config.N_SEC_FEATS)
    output = model(dummy_crime_input,dummy_crime_sec)
    
    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of trainable parameters: {n_total_params}')

        
            







