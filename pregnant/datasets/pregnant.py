from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import pickle
import numpy as np


class Pregnant(Dataset):
    def __init__(self, root, split):
        self.path = os.path.join(root, split+'.csv')
        self.data = pd.read_csv(self.path)
        self.data = self.reduce_mem_usage(self.data)
        self.target_cols = ['premature', 'low_BW', 'macrosomia', 'death', 'malformation']
        # self.target_cols = ['SGA_S1', 'LGA_S1', 'premature', 'low_BW',
        #     'macrosomia', 'death', 'malformation']
        self.train_cols = [x for x in self.data.columns if x not in self.target_cols]
        self.input = self.data.loc[:, self.train_cols].values
        self.label = self.data.loc[:, self.target_cols].values
        # 获取分类变量的idx及种类数量
        with open(os.path.join(root, 'cat_info.pkl'), 'rb') as file:
            self.categorical_columns_idxs, self.categorical_dims_idxs = pickle.load(file)

    def __getitem__(self, idx):
        x = self.input[idx]
        y = self.label[idx]

        # data transformer
        x = torch.from_numpy(x)
        y = torch.from_numpy(y).long()
        return x, y

    def __len__(self):
        return len(self.data)
    
    @property
    def get_categorical_columns_idxs(self):
        return self.categorical_columns_idxs
    
    @property
    def get_categorical_dims_idxs(self):
        return self.categorical_dims_idxs
    
    @property
    def get_train_cols(self):
        return self.train_cols
    
    @property
    def get_target_cols(self):
        return self.target_cols
    
    def reduce_mem_usage(self, df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2    
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)    
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df


if __name__ == '__main__':
    pregnantDataset = Pregnant('pregnant/data/bishe_data','toy')
    a, b = pregnantDataset.__getitem__(0)
    print(a.shape)
    print(b.shape)

    