import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

class Dataset_BTC(Dataset):
    def __init__(self, root_path, flag='train', size=None, target='pct_close'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'valid']
        self.set_flag = flag
        
        self.target = target
        
        self.root_path = root_path
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        self.__read_data__()

    def __read_data__(self):
        # BTC.train.csv, BTC.test.csv, BTC.val.csv
        data_path = "BTC.%s.csv" % self.set_flag
        df = pd.read_csv(os.path.join(self.root_path, data_path))

        # dfx = df.drop([self.target], axis=1)
        dfx = df
        dfy = df[self.target]

        self.scaler_x.fit(dfx.values)
        self.data_x = self.scaler_x.transform(dfx.values)
        self.scaler_y.fit(dfy.values)
        self.data_y = self.scaler_y.transform(dfy.values)
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y.reshape(-1, 1)
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler_y.inverse_transform(data)
