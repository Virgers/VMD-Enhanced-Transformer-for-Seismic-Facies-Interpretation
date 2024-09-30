import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')


class TSF_custom(Dataset):

    """
    Split dataset randomly first, then split the train, val, test(based on the whole dataset)
    return train data, val data, train label, val label, test data(test data without labels)
    """

    def __init__(self,  flag, root_path, data_path, label_path, mask_path, train_proportion, test_proportion, val_proportion):
       
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        self.label_path = label_path
        self.mask_path = mask_path
        self.train_proportion = train_proportion
        self.test_proportion = test_proportion
        self.val_proportion = val_proportion     
        self.__read_data__()

    def __read_data__(self):

        # data raw, label raw are all 3D with split_data_label function convert to 2D
        # since after we apply the vmd to seismic data the shape will turn into 254
       
        self.data_raw = np.load(os.path.join(self.root_path, self.data_path))
        self.label_raw = np.load(os.path.join(self.root_path, self.label_path))
        self.mask_raw = np.load(os.path.join(self.root_path, self.label_path))

        """if f3 use this"""
        data_raw = self.data_raw
        label_raw = self.label_raw
        mask_raw = self.mask_raw

        """if newzealand use this"""
        # data_raw = self.data_raw['data']
        # label_raw = self.label_raw['labels']
        # mask_raw = self.mask_raw['labels']
        # mask_raw = mask_raw.swapaxes(-1, 1)
        # attn_mask_raw = attn_mask_raw.swapaxes(-1, 1)

        split_datasets = self.split_data_label(data_raw, label_raw, mask_raw)

        if self.flag == 'train':
            self.train_data, self.train_label, self.train_mask = split_datasets['train']
            
        elif self.flag == 'val':
            self.val_data, self.val_label, self.val_mask = split_datasets['val']

        elif self.flag == 'test': 
            self.test_data, self.test_label, self.test_mask = split_datasets['test']


    def split_data_label(self, data_raw, label_raw, mask_raw):
        # 3d

        """ F3 """
        data_raw_shape = data_raw.shape
        label_raw_shape = label_raw.shape
        mask_raw_shape = mask_raw.shape

        label_raw_2D = label_raw.reshape(label_raw_shape[0] * label_raw_shape[1], label_raw_shape[2])
        mask_raw_2D = mask_raw.reshape(mask_raw_shape[0] * mask_raw_shape[1], mask_raw_shape[2])

        data_raw_for_scale = data_raw.reshape(-1, 255)
        self.scaler = StandardScaler()
        self.scaler.fit(data_raw_for_scale)
        data_raw_2D = self.scaler.transform(data_raw_for_scale)
        data_raw_2D = data_raw_2D.reshape(-1, 255)

        """ NZ """
        # label_raw_2D_swapaxes = np.swapaxes(label_raw['labels'], 0, 1).swapaxes(-1, 1)
        # label_raw_shape = label_raw_2D_swapaxes.shape
        # label_raw_2D = label_raw_2D_swapaxes.reshape(label_raw_shape[0] * label_raw_shape[1], label_raw_shape[2])
        
        # mask_raw_2D_swapaxes = np.swapaxes(mask_raw['labels'], 0, 1).swapaxes(-1, 1)
        # mask_raw_shape = mask_raw_2D_swapaxes.shape
        # mask_raw_2D = mask_raw_2D_swapaxes.reshape(mask_raw_shape[0] * mask_raw_shape[1], mask_raw_shape[2])

        # data_raw_2D_swapaxes = np.swapaxes(data_raw['data'], 0, 1).swapaxes(-1, 1)
        # data_raw_shape = data_raw_2D_swapaxes.shape
        # data_raw_2D = data_raw_2D_swapaxes.reshape(data_raw_shape[0] * data_raw_shape[1], data_raw_shape[2])

        data_raw_for_scale = data_raw_2D
        self.scaler = StandardScaler()
        self.scaler.fit(data_raw_for_scale)
        data_raw_2D = self.scaler.transform(data_raw_for_scale)
        

        np.random.seed(42)
        # Define proportions for each subset
        train_proportion = self.train_proportion
        test_proportion = self.test_proportion
        val_proportion = self.val_proportion
        
        
        # Calculate sizes of each subset
        num_samples = len(data_raw_2D)  
        train_size = int(train_proportion * num_samples) # 56220
        test_size = int(test_proportion * num_samples)   # 196770
        val_size =  int(val_proportion * num_samples) # 28110
        
        
        train_indices = np.random.choice(num_samples, size=train_size, replace=False)
        # test_indices = np.random.choice(num_samples, size=test_size, replace=False)
        val_indices = np.random.choice(num_samples, size=val_size, replace=False)
        

        # Select subsets of the dataset using the random indices
        "we iterate the train/ val/ valtest by randomly generated indices"
        train_data = data_raw_2D[train_indices]
        train_label = label_raw_2D[train_indices]
        train_mask = mask_raw_2D[train_indices]

        val_data = data_raw_2D[val_indices]
        val_label = label_raw_2D[val_indices]
        val_mask = mask_raw_2D[val_indices]
              
        test_data = data_raw_2D[:test_size]
        test_label = label_raw_2D[:test_size]
        test_mask = mask_raw_2D[:test_size]
    
        train_data = train_data[:, :, np.newaxis]
        train_mask = train_mask[:, :, np.newaxis]
        train_label = train_label[:, :, np.newaxis]
        
        val_data = val_data[:, :, np.newaxis]
        val_mask = val_mask[:, :, np.newaxis]
        val_label = val_label[:, :, np.newaxis]
        
        test_data = test_data[:, :, np.newaxis]
        test_mask = test_mask[:, :, np.newaxis]
        test_label = test_label[:, :, np.newaxis]
    
        datasets = {'train': (train_data, train_label, train_mask),
                'val': (val_data, val_label, val_mask),
                'test': (test_data, test_label, test_mask)}
        
        return datasets
    
    def __getitem__(self, ind):

        if self.flag == 'train':
            return torch.Tensor(self.train_data[ind]), torch.Tensor(self.train_label[ind]),  torch.Tensor(self.train_mask[ind])
        elif self.flag == 'val':
            return torch.Tensor(self.val_data[ind]), torch.Tensor(self.val_label[ind]),  torch.Tensor(self.val_mask[ind])

        elif self.flag == 'test':
            return torch.Tensor(self.test_data[ind]), torch.Tensor(self.test_label[ind]),  torch.Tensor(self.test_mask[ind])
        

    
    def __len__(self):
        if self.flag == 'train':
            return (self.train_data.shape[0])
        if self.flag == 'val':
            return (self.val_data.shape[0])
        if self.flag == 'test':
         return (self.test_data.shape[0])
   