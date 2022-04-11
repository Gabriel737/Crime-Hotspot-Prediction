#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 19:20:43 2022

@author: arshdeepsingh
"""
import torch
import torch.nn as nn

import config
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
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

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
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
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
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

        self.conv3d = nn.Conv3d(in_channels=hidden_dim,out_channels=1,kernel_size=(1,3,3),padding=(0,1,1),bias=True)
    
    def forward(self, input, hidden_state=None):
        out = self.convlstm1(input)
        # print(f'conv output: {out.shape}')
        out = out.permute(0,2,1,3,4)
        out = self.batch_norm1(out)
        # print(f'bn output: {out.shape}')
        out = self.maxpool(out)
        # print(f'maxpool output: {out.shape}')
        out = self.dropout(out)
        # print(f'dropout output: {out.shape}')

        out = out.permute(0,2,1,3,4)
        # print(out.shape)
        out = self.convlstm2(out)
        # print(f'conv output: {out.shape}')
        out = out.permute(0,2,1,3,4)
        out = self.batch_norm2(out)
        # print(f'bn output: {out.shape}')
        out = self.maxpool(out)
        # print(f'maxpool output: {out.shape}')
        out = self.dropout(out)
        # print(f'dropout output: {out.shape}')

        out = out.permute(0,2,1,3,4)
        out = self.convlstm3(out)
        # print(f'conv output: {out.shape}')
        out = out.permute(0,2,1,3,4)
        out = self.batch_norm3(out)
        # print(f'bn output: {out.shape}')
        out = self.maxpool(out)
        # print(f'maxpool output: {out.shape}')
        out = self.dropout(out)
        # print(f'dropout output: {out.shape}')

        out = out.permute(0,2,1,3,4)
        out = self.convlstm4(out)
        # print(f'conv output: {out.shape}')
        out = out.permute(0,2,1,3,4)
        out = self.batch_norm4(out)
        # print(f'bn output: {out.shape}')
        out = self.maxpool(out)
        # print(f'maxpool output: {out.shape}')
        out = self.dropout(out)
        # print(f'dropout output: {out.shape}')

        out = self.conv3d(out)
        # print(f'conv3d output: {out.shape}')
        out = self.sigmoid(out)
        return out
