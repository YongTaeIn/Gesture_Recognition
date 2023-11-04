from torch.utils.data import Dataset,DataLoader,random_split
import torch
import os
import pandas as pd
import numpy as np

class CustomDataset(Dataset):
    
    # 데이터 전처리
    def __init__(self, window_size,Folder_dir):
        self.data_list=[]
        swindow_size=window_size
        Folder_path=Folder_dir

        for filename in os.listdir(Folder_path):

            data=pd.read_csv(Folder_path+filename)
            data=data.to_numpy()



            features=torch.FloatTensor(data[:,0:99])
            labels=torch.FloatTensor(data[:,99])
            labels=labels.reshape(len(labels),-1)
            
            
            for i in range(len(features)-window_size):
                features_subset=features[i:i+window_size]
                labels_subset=labels[i]
                
                # print(features_subset.shape)
                # print(labels_subset.shape)
                self.data_list.append([features_subset,labels_subset])

    # 데이터 길이
    def __len__(self):

        return len(self.data_list)


    # 데이터 셋을 전처리 후 반환하는 함수
    def __getitem__(self, idx):
        
        x,y=self.data_list[idx]

        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        return x, y