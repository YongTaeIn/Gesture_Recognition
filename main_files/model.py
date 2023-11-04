## torch 모듈에서는 Super().__init__() 을 해줘야 한다. 

import torch.nn as nn  # Neural Network, activation function 모듈의 기본 클래스
import torch

class CNN_LSTM(nn.Module):
    def __init__(self,input_size,output_size,units):
        super(CNN_LSTM,self).__init__()
        
        self.conv1d=nn.Conv1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=3,
            stride=1,
            padding=0
        )

        self.lstm=nn.LSTM(
            input_size=64,
            hidden_size=units,
            num_layers=1,
            dropout=0.2,
            batch_first=True) # if True = (batch, seq, features)
        
        self.Relu=nn.ReLU()

        self.Linear_1=nn.Linear(
            in_features=32,
            out_features=4
        )

        self.softmax=nn.Softmax()

    def forward(self,x):
        # x.shape=(batch_size,seq_len,feautres) 
        #          -> (batch_size,feautres,seq_len)
        x=x.permute(0,2,1)
        x=self.conv1d(x)
        x=self.Relu(x)


        x=x.permute(0,2,1)  # torch.Size([64,28,64])
        h_n, c_n=self.lstm(x)

        x=self.Linear_1(h_n[:,-1,:])

        x=self.softmax(x)
        return x
