import os
import numpy as np
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

    def __init__(self, is_vmd, flag, root_path, data_path, vmd_data_path, label_path,  mask_path, train_proportion, test_proportion, val_proportion):
        self.is_vmd = is_vmd
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        
        self.vmd_data_path = vmd_data_path
        self.label_path = label_path
        
        self.mask_path = mask_path
        self.train_proportion = train_proportion
        self.test_proportion = test_proportion
        self.val_proportion = val_proportion
       
        self.__read_data__()

    def __read_data__(self):

        if self.is_vmd:
            self.data_raw = np.load(os.path.join(self.root_path, self.vmd_data_path))
        else:
            self.data_raw = np.load(os.path.join(self.root_path, self.data_path))

        self.label_raw = np.load(os.path.join(self.root_path, self.label_path))
        self.mask_raw = self.attn_mask = self.label_raw
        
        if not self.is_vmd :

            data_raw = self.data_raw['data']
            data_raw = np.swapaxes(data_raw, 1, 0)
            data_raw = np.swapaxes(data_raw, -1, 1) 

            label_raw = self.label_raw['labels']
            # make sure labels start from 0
            label_raw = label_raw - 1
            label_raw = np.swapaxes(label_raw, 1, 0)
            label_raw = np.swapaxes(label_raw, -1, 1) 

            mask_raw = label_raw

        else:
            data_raw = self.data_raw
            data_raw = np.swapaxes(data_raw, -1, 1)  # (-1, 8, 1006) --> (-1, 1006, 8)
                                                        
            label_raw = self.label_raw['labels']
            label_raw = label_raw - 1
            label_raw = np.swapaxes(label_raw, 1, 0)
            label_raw = np.swapaxes(label_raw, -1, 1)

            mask_raw = label_raw
        """if we choose the nz we do not need swap axis"""
        
        split_datasets = self.split_data_label(data_raw, label_raw, mask_raw, self.is_vmd)


        if self.flag == 'train':
            self.train_data, self.train_label, self.train_mask = split_datasets['train']
            
        elif self.flag == 'val':
            self.val_data, self.val_label, self.val_mask = split_datasets['val']

        elif self.flag == 'test': # one can just use the test data while left the test_label
            self.test_data, self.test_label, self.test_mask = split_datasets['test']

    def split_data_label(self, data_raw, label_raw, mask_raw, is_vmd):

        label_raw_shape = label_raw.shape
        mask_raw_shape = mask_raw.shape
        
        label_raw_2D = label_raw.reshape(label_raw_shape[0] * label_raw_shape[1], label_raw_shape[2])
        # mask_raw_2D = label_raw_2D
        mask_raw_2D = label_raw.reshape(mask_raw_shape[0] * mask_raw_shape[1], mask_raw_shape[2])
        
        if is_vmd:
            data_raw_2D = data_raw.swapaxes(-1, 1)
            
            """for nz dataset reshape size depends on the vmd dataset original size"""
            self.scaler = StandardScaler()
            data_raw_for_scale = data_raw_2D.reshape(-1, 1006)
            self.scaler.fit(data_raw_for_scale)
            data_raw_2D = self.scaler.transform(data_raw_for_scale)
            data_raw_2D = data_raw_2D.reshape(-1, 8, 1006)
            data_raw_2D = np.swapaxes(data_raw_2D, -1, 1)    

        else:
            data_raw_for_scale = data_raw.reshape(-1, 1006)

            self.scaler = StandardScaler()
            self.scaler.fit(data_raw_for_scale)
            data_raw_2D = self.scaler.transform(data_raw_for_scale)
            data_raw_2D = data_raw_2D.reshape(-1, 1006)

        np.random.seed(42)

        # Define proportions for each subset
        train_proportion = self.train_proportion
        test_proportion = self.test_proportion
        val_proportion = self.val_proportion
       

        # Calculate sizes of each subset
        num_samples = len(data_raw_2D)  
        train_size = int(train_proportion * num_samples) 
        test_size = int(test_proportion * num_samples)   
        val_size =  int(val_proportion * num_samples) 
      
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
        
        # test first test_size dataset
        test_data = data_raw_2D[:test_size]
        test_label = label_raw_2D[:test_size]
        test_mask = mask_raw_2D[:test_size]
       
        # Expand a new axis here, because we are only 1 features , otherwise collate_fn mask will not work
        # if we take the vmd code we do not need the additional axies for collate fn vise versa

        if not is_vmd:
            train_data = train_data[:, :, np.newaxis]
            val_data = val_data[:, :, np.newaxis]
            test_data = test_data[:, :, np.newaxis]
            
        train_label = train_label[:, :, np.newaxis]
        train_mask = train_mask[:, :, np.newaxis]
    
        val_label = val_label[:, :, np.newaxis]
        val_mask = val_mask[:, :, np.newaxis]
  

        test_label = test_label[:, :, np.newaxis]
        test_mask = test_mask[:, :, np.newaxis]
        

        datasets = {'train': (train_data, train_label, train_mask),
                'val': (val_data, val_label, val_mask),
                'test': (test_data, test_label, test_mask),
                }
     
        
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
     