#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MITBIH.py
PyTorch dataloaders for MITHIB dataset
Author: Xiaomin Li, Texas State University
Date: 1/26/2023
TODOS:
* 
"""


#necessory import libraries

import os 
import sys 
import numpy as np
import pandas as pd
from tqdm import tqdm 

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
from sklearn.utils.random import sample_without_replacement

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#class names and corresponding labels of the MITBIH dataset
cls_dit = {'Non-Ectopic Beats':0, 'Superventrical Ectopic':1, 'Ventricular Beats':2,
                                                'Unknown':3, 'Fusion Beats':4}
reverse_cls_dit = {0:'Non-Ectopic Beats', 1:'Superventrical Ectopic', 2:'Ventricular Beats', 3: 'Unknown', 4:'Fusion Beats'}



class mitbih_oneClass(Dataset):
    """
    A pytorch dataloader loads on class data from mithib_train dataset.
    Example Usage:
        class0 = mitbih_oneClass(class_id = 0)
        class1 = mitbih_oneClass(class_id = 1)
    """
    def __init__(self, filename='./mitbih_train.csv', reshape = True, class_id = 0):
        data_pd = pd.read_csv(filename, header=None)
        data = data_pd[data_pd[187] == class_id]
    
        self.data = data.iloc[:, :128].values  # change it to only has 128 timesteps to match conv1d dim changes
        self.labels = data[187].values
        
        if reshape:
            self.data = self.data.reshape(self.data.shape[0], 1, self.data.shape[1])
        
        print(f'Data shape of {reverse_cls_dit[class_id]} instances = {self.data.shape}')
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        
class mitbih_twoClass(Dataset):
    """
    A pytorch dataloader loads two class data from mithib_train dataset.
    Example Usage:
        class0_1 = mitbih_twoClass(class_id1 = 0, class_id2 = 1)
        class1_2 = mitbih_twoClass(class_id1 = 1, class_id2 = 2)
    """
    def __init__(self, filename='./mitbih_train.csv', reshape = True, class_id1 = 0, class_id2 = 1):
        data_pd = pd.read_csv(filename, header=None)
        data_1 = data_pd[data_pd[187] == class_id1]
        data_2 = data_pd[data_pd[187] == class_id2]
    
        self.data_1 = data_1.iloc[:, :-1].values
        self.labels_1 = data_1[187].values
        
        self.data_2 = data_2.iloc[:, :-1].values
        self.labels_2 = data_2[187].values
        
        if reshape:
            self.data_1 = self.data_1.reshape(self.data_1.shape[0], 1, 1, self.data_1.shape[1])
            self.data_2 = self.data_2.reshape(self.data_2.shape[0], 1, 1, self.data_2.shape[1])
        
        print(f'Data shape of {reverse_cls_dit[class_id1]} instances = {self.data_1.shape}')
        print(f'Data shape of {reverse_cls_dit[class_id2]} instances = {self.data_2.shape}')
        
    def __len__(self):
        return min(len(self.labels_1), len(self.labels_2))
    
    def __getitem__(self, idx):
        return self.data_1[idx], self.labels_1[idx], self.data_2[idx], self.labels_2[idx]

    
    
class mitbih_allClass(Dataset):
    def __init__(self, filename='./mitbih_train.csv', isBalanced = True, n_samples=20000, oneD=True):
        data_train = pd.read_csv(filename, header=None)
        
        # making the class labels for our dataset
        data_0 = data_train[data_train[187] == 0]
        data_1 = data_train[data_train[187] == 1]
        data_2 = data_train[data_train[187] == 2]
        data_3 = data_train[data_train[187] == 3]
        data_4 = data_train[data_train[187] == 4]
        
        if isBalanced:
            data_0_resample = resample(data_0, n_samples=n_samples, 
                               random_state=123, replace=True)
            data_1_resample = resample(data_1, n_samples=n_samples, 
                                       random_state=123, replace=True)
            data_2_resample = resample(data_2, n_samples=n_samples, 
                                       random_state=123, replace=True)
            data_3_resample = resample(data_3, n_samples=n_samples, 
                                       random_state=123, replace=True)
            data_4_resample = resample(data_4, n_samples=n_samples, 
                                       random_state=123, replace=True)

            train_dataset = pd.concat((data_0_resample, data_1_resample, 
                                      data_2_resample, data_3_resample, data_4_resample))
        else:
            train_dataset = pd.concat((data_0, data_1, 
                                      data_2, data_3, data_4))

        self.X_train = train_dataset.iloc[:, :128].values
        if oneD:
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])
        else:
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, 1, self.X_train.shape[1])
        self.y_train = train_dataset[187].values
            
        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        if isBalanced:
            print(f'The dataset including {len(data_0_resample)} class 0, {len(data_1_resample)} class 1, \
                  {len(data_2_resample)} class 2, {len(data_3_resample)} class 3, {len(data_4_resample)} class 4')
        else:
            print(f'The dataset including {len(data_0)} class 0, {len(data_1)} class 1, {len(data_2)} class 2, {len(data_3)} class 3, {len(data_4)} class 4')
        
        
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
    
    
class mitbih_augData(Dataset):
    def __init__(self, filename='./mitbih_train.csv', n_samples=20000, target_class = 0):
        data_train = pd.read_csv(filename, header=None)
        
        # making the class labels for our dataset
        data_0 = data_train[data_train[187] == 0].iloc[:, :-1].values
        data_0 = np.expand_dims(data_0, axis=1)
        data_1 = data_train[data_train[187] == 1].iloc[:, :-1].values
        data_1 = np.expand_dims(data_1, axis=1)
        data_2 = data_train[data_train[187] == 2].iloc[:, :-1].values
        data_2 = np.expand_dims(data_2, axis=1)
        data_3 = data_train[data_train[187] == 3].iloc[:, :-1].values
        data_3 = np.expand_dims(data_3, axis=1)
        data_4 = data_train[data_train[187] == 4].iloc[:, :-1].values
        data_4 = np.expand_dims(data_4, axis=1)
        
        data_0_resample = resample(data_0, n_samples=n_samples, 
                           random_state=123, replace=True)
        data_1_resample = resample(data_1, n_samples=n_samples, 
                                   random_state=123, replace=True)
        data_2_resample = resample(data_2, n_samples=n_samples, 
                                   random_state=123, replace=True)
        data_3_resample = resample(data_3, n_samples=n_samples, 
                                   random_state=123, replace=True)
        data_4_resample = resample(data_4, n_samples=n_samples, 
                                   random_state=123, replace=True)

        
        org_dataset = np.concatenate((data_0_resample, data_1_resample, 
                                      data_2_resample, data_3_resample, data_4_resample), axis=1)
        print(org_dataset.shape) #[n_samples, 5, 187]
        

        self.org_train = np.expand_dims(org_dataset, axis=1)
        print(f'org_train shape is {self.org_train.shape}') #[n_samples, 1, 5, 187]
        
        tag_data = data_train[data_train[187] == target_class].iloc[:, :-1].values
        tag_data = np.expand_dims(tag_data, axis=1)
        tag_data_resample = resample(tag_data, n_samples=n_samples, 
                           random_state=123, replace=True)
        self.tag_train = np.expand_dims(tag_data_resample, axis=1)
        print(f'tag_train shape is {self.tag_train.shape}') #[n_samples, 1, 1, 187]
        
    def __len__(self):
        return len(self.tag_train)
    def __getitem__(self, idx):
        return self.org_train[idx], self.tag_train[idx]

def main():
    class0 = mitbih_oneClass(class_id = 0)
    class_0_1 = mitbih_twoClass(class_id1 = 0, class_id2 = 1)
    data = mitbih_allClass(isBalanced = True, n_samples=2000)
#     data = mitbih_allClass(isBalanced = False)
    
#     mixData = mitbih_augData(n_samples=2000)
#     for i, (org_data, tag_data) in enumerate(mixData):
#         print(org_data.shape)
#         print(tag_data.shape)
#         if i > 3:
#             break
    pass
        
if __name__ == "__main__":
    main()