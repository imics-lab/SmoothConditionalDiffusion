# -*- coding: utf-8 -*-
"""UniMiB_SHAR_ADL_load_dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1U1EY6cZsOFERD3Df1HRqjuTq5bDUGH03

#UniMiB_SHAR_ADL_load_dataset.ipynb. 
Loads the A-9 (ADL) portion of the UniMiB dataset from the Internet repository and converts the data into numpy arrays while adhering to the general format of the [Keras MNIST load_data function](https://keras.io/api/datasets/mnist/#load_data-function).

Arguments: tbd
Returns: Tuple of Numpy arrays:   
(x_train, y_train),(x_validation, y_validation)\[optional\],(x_test, y_test) 

* x_train\/validation\/test: containing float64 with shapes (num_samples, 151, {3,4,1})
* y_train\/validation\/test: containing int8 with shapes (num_samples 0-9)

The train/test split is by subject

Example usage:  
x_train, y_train, x_test, y_test = unimib_load_dataset()

Additional References  
 If you use the dataset and/or code, please cite this paper (downloadable from [here](http://www.mdpi.com/2076-3417/7/10/1101/html))

Developed and tested using colab.research.google.com  
To save as .py version use File > Download .py

Author:  Lee B. Hinkle, IMICS Lab, Texas State University, 2021

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.


TODOs:
* Fix document strings
* Assign names to activities instead of numbers
"""

import os
import shutil #https://docs.python.org/3/library/shutil.html
from shutil import unpack_archive # to unzip
#from shutil import make_archive # to create zip for storage
import requests #for downloading zip file
from scipy import io #for loadmat, matlab conversion
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt # for plotting - pandas uses matplotlib
from tabulate import tabulate # for verbose tables
#from tensorflow.keras.utils import to_categorical # for one-hot encoding

#credit https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
#many other methods I tried failed to download the file properly
from torch.utils.data import Dataset, DataLoader
import torch

#data augmentation
import tsaug

from time import gmtime, strftime, localtime #for displaying Linux UTC timestamps in hh:mm:ss
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class_dict = {'StandingUpFS':0,'StandingUpFL':1,'Walking':2,'Running':3,'GoingUpS':4,'Jumping':5,'GoingDownS':6,'LyingDownFS':7,'SittingDown':8}

class unimib_load_dataset(Dataset):
    def __init__(self, 
        verbose = False,
        incl_xyz_accel = False, #include component accel_x/y/z in ____X data
        incl_rms_accel = True, #add rms value (total accel) of accel_x/y/z in ____X data
        incl_val_group = False, #True => returns x/y_test, x/y_validation, x/y_train
                               #False => combine test & validation groups
        is_normalize = False,
        split_subj = dict
                    (train_subj = [4,5,6,7,8,10,11,12,14,15,19,20,21,22,24,26,27,29],
                    validation_subj = [1,9,16,23,25,28],
                    test_subj = [2,3,13,17,18,30]),
        one_hot_encode = True, data_mode = 'Train', single_class = False, class_name= 'Walking', augment_times = None):
        
        self.verbose = verbose
        self.incl_xyz_accel = incl_xyz_accel
        self.incl_rms_accel = incl_rms_accel
        self.incl_val_group = incl_val_group
        self.split_subj = split_subj
        self.one_hot_encode = one_hot_encode
        self.data_mode = data_mode
        self.class_name = class_name
        self.single_class = single_class
        self.is_normalize = is_normalize
        
        
        #Download and unzip original dataset
        if (not os.path.isfile('./UniMiB-SHAR.zip')):
            print("Downloading UniMiB-SHAR.zip file")
            #invoking the shell command fails when exported to .py file
            #redirect link https://www.dropbox.com/s/raw/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip
            #!wget https://www.dropbox.com/s/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip
            self.download_url('https://www.dropbox.com/s/raw/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip','./UniMiB-SHAR.zip')
        if (not os.path.isdir('./UniMiB-SHAR')):
            shutil.unpack_archive('./UniMiB-SHAR.zip','.','zip')
        #Convert .mat files to numpy ndarrays
        path_in = './UniMiB-SHAR/data'
        #loadmat loads matlab files as dictionary, keys: header, version, globals, data
        adl_data = io.loadmat(path_in + '/adl_data.mat')['adl_data']
        adl_names = io.loadmat(path_in + '/adl_names.mat', chars_as_strings=True)['adl_names']
        adl_labels = io.loadmat(path_in + '/adl_labels.mat')['adl_labels']

        if(self.verbose):
            headers = ("Raw data","shape", "object type", "data type")
            mydata = [("adl_data:", adl_data.shape, type(adl_data), adl_data.dtype),
                    ("adl_labels:", adl_labels.shape ,type(adl_labels), adl_labels.dtype),
                    ("adl_names:", adl_names.shape, type(adl_names), adl_names.dtype)]
            print(tabulate(mydata, headers=headers))
        #Reshape data and compute total (rms) acceleration
        num_samples = 151 
        #UniMiB SHAR has fixed size of 453 which is 151 accelX, 151 accely, 151 accelz
        adl_data = np.reshape(adl_data,(-1,num_samples,3), order='F') #uses Fortran order
        if (self.incl_rms_accel):
            rms_accel = np.sqrt((adl_data[:,:,0]**2) + (adl_data[:,:,1]**2) + (adl_data[:,:,2]**2))
            adl_data = np.dstack((adl_data,rms_accel))
        #remove component accel if needed
        if (not self.incl_xyz_accel):
            adl_data = np.delete(adl_data, [0,1,2], 2)
        if(verbose):
            headers = ("Reshaped data","shape", "object type", "data type")
            mydata = [("adl_data:", adl_data.shape, type(adl_data), adl_data.dtype),
                    ("adl_labels:", adl_labels.shape ,type(adl_labels), adl_labels.dtype),
                    ("adl_names:", adl_names.shape, type(adl_names), adl_names.dtype)]
            print(tabulate(mydata, headers=headers))
        #Split train/test sets, combine or make separate validation set
        #ref for this numpy gymnastics - find index of matching subject to sub_train/sub_test/sub_validate
        #https://numpy.org/doc/stable/reference/generated/numpy.isin.html


        act_num = (adl_labels[:,0])-1 #matlab source was 1 indexed, change to 0 indexed
        sub_num = (adl_labels[:,1]) #subject numbers are in column 1 of labels

        if (not self.incl_val_group):
            train_index = np.nonzero(np.isin(sub_num, self.split_subj['train_subj'] + 
                                            self.split_subj['validation_subj']))
            x_train = adl_data[train_index]
            y_train = act_num[train_index]
        else:
            train_index = np.nonzero(np.isin(sub_num, self.split_subj['train_subj']))
            x_train = adl_data[train_index]
            y_train = act_num[train_index]

            validation_index = np.nonzero(np.isin(sub_num, self.split_subj['validation_subj']))
            x_validation = adl_data[validation_index]
            y_validation = act_num[validation_index]

        test_index = np.nonzero(np.isin(sub_num, self.split_subj['test_subj']))
        x_test = adl_data[test_index]
        y_test = act_num[test_index]

        if (verbose):
            print("x/y_train shape ",x_train.shape,y_train.shape)
            if (self.incl_val_group):
                print("x/y_validation shape ",x_validation.shape,y_validation.shape)
            print("x/y_test shape  ",x_test.shape,y_test.shape)
        #If selected one-hot encode y_* using keras to_categorical, reference:
        #https://keras.io/api/utils/python_utils/#to_categorical-function and
        #https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
        if (self.one_hot_encode):
            y_train = self.to_categorical(y_train, num_classes=9)
            if (self.incl_val_group):
                y_validation = self.to_categorical(y_validation, num_classes=9)
            y_test = self.to_categorical(y_test, num_classes=9)
            if (verbose):
                print("After one-hot encoding")
                print("x/y_train shape ",x_train.shape,y_train.shape)
                if (self.incl_val_group):
                    print("x/y_validation shape ",x_validation.shape,y_validation.shape)
                print("x/y_test shape  ",x_test.shape,y_test.shape)
#         if (self.incl_val_group):
#             return x_train, y_train, x_validation, y_validation, x_test, y_test
#         else:
#             return x_train, y_train, x_test, y_test
        
        # reshape x_train, x_test data shape from (BH, length, channel) to (BH, channel, 1, length)
        self.x_train = np.transpose(x_train, (0, 2, 1))
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1, self.x_train.shape[2])
        self.x_train = self.x_train[:,:,:,:-1]
        self.y_train = y_train
        
        self.x_test = np.transpose(x_test, (0, 2, 1))
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1, self.x_test.shape[2])
        self.x_test = self.x_test[:,:,:,:-1]
        self.y_test = y_test
        print(f'x_train shape is {self.x_train.shape}, x_test shape is {self.x_test.shape}')
        print(f'y_train shape is {self.y_train.shape}, y_test shape is {self.y_test.shape}')
        
        
        if self.is_normalize:
            self.x_train = self.normalization(self.x_train)
            self.x_test = self.normalization(self.x_test)
        
        #Return the give class train/test data & labels
        if self.single_class:
            one_class_train_data = []
            one_class_train_labels = []
            one_class_test_data = []
            one_class_test_labels = []

            for i, label in enumerate(y_train):
                if label == class_dict[self.class_name]:
                    one_class_train_data.append(self.x_train[i])
                    one_class_train_labels.append(label)

            for i, label in enumerate(y_test):
                if label == class_dict[self.class_name]:
                    one_class_test_data.append(self.x_test[i])
                    one_class_test_labels.append(label)
            self.one_class_train_data = np.array(one_class_train_data)
            self.one_class_train_labels = np.array(one_class_train_labels)
            self.one_class_test_data = np.array(one_class_test_data)
            self.one_class_test_labels = np.array(one_class_test_labels)
            
            if augment_times:
                augment_data = []
                augment_labels = []
                for data, label in zip(one_class_train_data, one_class_train_labels):
#                     print(data.shape) # C, 1, T
                    data = data.reshape(data.shape[0], data.shape[2]) # Channel, Timestep
                    data = np.transpose(data, (1, 0)) # T, C
                    data = np.asarray(data)
                    for i in range(augment_times):
                    
                        aug_data = tsaug.Quantize(n_levels=[10, 20, 30]).augment(data)
                        aug_data = tsaug.Drift(max_drift=(0.1, 0.5)).augment(aug_data)                   
#                         aug_data = my_augmenter(data) # T, C 
                        aug_data = np.transpose(aug_data, (1, 0)) # C, T
                        aug_data = aug_data.reshape(aug_data.shape[0], 1, aug_data.shape[1]) # C, 1, T
                        augment_data.append(aug_data)
                        augment_labels.append(label)
                
                augment_data = np.array(augment_data)
                augment_labels = np.array(augment_labels)
                print(f'augment_data shape is {augment_data.shape}')
                print(f'augment_labels shape is {augment_labels.shape}')
                self.one_class_train_data = np.concatenate((augment_data, self.one_class_train_data), axis = 0)
                self.one_class_train_labels = np.concatenate((augment_labels, self.one_class_train_labels), axis = 0)
            
            print(f'return single class data and labels, class is {self.class_name}')
            print(f'train_data shape is {self.one_class_train_data.shape}, test_data shape is {self.one_class_test_data.shape}')
            print(f'train label shape is {self.one_class_train_labels.shape}, test data shape is {self.one_class_test_labels.shape}')
        
    def download_url(self, url, save_path, chunk_size=128):
        r = requests.get(url, stream=True)
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]       

    
    def _normalize(self, epoch):
        """ A helper method for the normalization method.
            Returns
                result: a normalized epoch
        """
        e = 1e-10
        result = (epoch - epoch.mean(axis=0)) / ((np.sqrt(epoch.var(axis=0)))+e)
        return result
    
    def _min_max_normalize(self, epoch):
        
        result = (epoch - min(epoch)) / (max(epoch) - min(epoch))
        return result

    def normalization(self, epochs):
        """ Normalizes each epoch e s.t mean(e) = 0 and var(e) = 1
            Args:
                epochs - Numpy structure of epochs
            Returns:
                epochs_n - mne data structure of normalized epochs (mean=0, var=1)
        """
        for i in range(epochs.shape[0]):
            for j in range(epochs.shape[1]):
                epochs[i,j,0,:] = self._normalize(epochs[i,j,0,:])
#                 epochs[i,j,0,:] = self._min_max_normalize(epochs[i,j,0,:])

        return epochs
    
    def __len__(self):
        
        if self.data_mode == 'Train':
            if self.single_class:
                return len(self.one_class_train_labels)
            else:
                return len(self.y_train)
        else:
            if self.single_class:
                return len(self.one_class_test_labels)
            else:
                return len(self.y_test)
        
    def __getitem__(self, idx):
        if self.data_mode == 'Train':
            if self.single_class:
                return self.one_class_train_data[idx], self.one_class_train_labels[idx]
            else:
                return self.x_train[idx], self.y_train[idx]
        else:
            if self.single_class:
                return self.one_class_test_data[idx], self.one_class_test_labels[idx]
            else:
                return self.x_test[idx], self.y_test[idx]
            
# # use get_x_y_sub to get partially processed numpy arrays
# full_filename = my_path+os.path.join('/HAR/e4_wristband_Nov2019/'+'e4_get_x_y_sub.py')
# shutil.copy(full_filename,'e4_get_x_y_sub.py')

#credit https://stackoverflow.com/users/4944093/george-petrov for name method
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]
def get_shapes(np_arr_list):
    """Returns text, each line is shape and dtype for numpy array in list
       example: print(get_shapes([X_train, X_test, y_train, y_test]))"""
    shapes = ""
    # shapes += "shapes call broke when making the function - not sure why"

    for i in np_arr_list:
        print('i=',i)
        my_name = namestr(i,globals())
        shapes += (my_name[0] + " shape is " + str(i.shape) \
            + " data type is " + str(i.dtype) + "\n")
    return shapes

def unzip_e4_file(zip_ffname, working_dir):
    """checks for local copy, if none unzips the e4 zipfile in working_dir
    Note:  the files themselves do not contain subject info and there are
    multiple files e.g. ACC.csv, BVP,csv etc, in each zipfile.
    It is very important to further process the files using _info.csv method
    :param zip_ffname: the path and filename of the zip file
    :param working_dir: local (colab) directory where csv files will be placed
    :return: nothing"""
    if (os.path.isdir(working_dir)):
        print("Skipping Unzip - Found existing archive in colab at", working_dir)
        return
    else:
        #print("Using source file", zip_ffname)
        print("Unzipping e4 file in local directory", working_dir)
        os.mkdir(working_dir)
        if (os.path.exists(zip_ffname)):
            shutil.unpack_archive(zip_ffname,working_dir,'zip')
        else:
            os.system('pwd')
            print("Error: ", zip_ffname, " not found, exiting")
            return

def df_from_e4_csv (ffname,col_labels):
    """"reads e4 csv file, uses start time and sample rate to create time indexed
    pandas dataframe.  Note only tested with ACC files, e4 csv files do not
    have header info, this will need to be added based on file type.
    :param ffname:  full filename e.g./content/temp/ACC.csv
    :col_labels: list of colums in csv - varies by type ['accel_x','accel_y...]
    :returns df: time indexed dataframe"""

    df = pd.read_csv(ffname, header=None)
    #start_time = pd.to_datetime(df.iloc[0,0]). #pain to convert too early
    #time.time() supports number of seconds which is what this is
    start_time = df.iloc[0,0] # first line in e4 csv
    sample_freq = df.iloc[1,0] # second line in e4 csv
    df = df.drop(df.index[[0,1]]) # drop 1st two rows, index is now off by 2
    print(ffname, "Sample frequency = ", sample_freq, " Hz")
    #show time in day month format, assumes same timezone
    print("File start time = ", strftime("%a, %d %b %Y %H:%M:%S", localtime(start_time)))  
    # Make the index datetime first so code can be used for other data types
    # Having the index as datetime is required for pandas resampling
    # The start_time pulled from the e4 csv file is a float64 which represents the
    # number of seconds since January 1, 1970, 00:00:00 (UTC)
    # UTC_time is computed for each row, then made into required datetime format
    # that pandas will accept as an index
    df['UTC_time'] = (df.index-2)/sample_freq + start_time
    end_time = df['UTC_time'].iloc[-1]
    print("File end time   = ",strftime("%a, %d %b %Y %H:%M:%S", localtime(end_time)))
    df['datetime'] = pd.to_datetime(df['UTC_time'], unit='s')
    df.set_index('datetime',inplace=True)
    df = df.drop('UTC_time', axis=1)
    df.columns = col_labels
    return df

def process_e4_accel(df):
    """converts component accel into g and adds accel_ttl column
    per info.txt range is [-2g, 2g] and unit in this file is 1/64g.
    """
    df['accel_x'] = df['accel_x']/64
    df['accel_y'] = df['accel_y']/64
    df['accel_z'] = df['accel_z']/64
    df_sqd = df.pow(2)[['accel_x', 'accel_y', 'accel_z']] #square each accel
    df_sum = df_sqd.sum(axis=1) #add sum of squares, new 1 col df
    df.loc[:,'accel_ttl'] = df_sum.pow(0.5)-1  # sqrt and remove 1g due to gravity
    del df_sqd, df_sum
    return df

def show_tag_time(tag_ffname):
    """utility prints time marks from tags.csv to help with video sync 
    and labeling.   When this is run in colab it seems to be GMT regardless
    of timezone settings."
    :param tag_ffname: file to be processed e.g. /content/temp/tags.csv'
    :return: nothing"""
    df_temp = pd.read_csv(tag_ffname, header=None)
    df_temp.columns = ['UTC_time']
    print ("    UTC_time          Local Time")
    for index, row in df_temp.iterrows():
        print(index, row['UTC_time'],
            strftime("%a, %d %b %Y %H:%M:%S", localtime(row['UTC_time'])))
# https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
# link to string formats for date and time

def label_df_from_csv (df, labels_ffname):
    """adds activity label and subject number to the dataframe based on the
    contents of a .csv file containing time and label info.
    Example csv format (see e4_time_sync.xlsx to help build csv from video)
        start,finish,label,sub
        2019:11:24 18:49:51,2019:11:24 18:50:18,Upstairs,1
        2019:11:24 18:50:18,2019:11:24 18:50:45,Downstairs,1
    :param df : time indexed dataframe from df_from_e4_csv method
    :labels_ffname : csv file with metadata
    :return : a dataframe with label and subject columns added"""
    df_labels = pd.read_csv(labels_ffname)
    df_labels['start'] =  pd.to_datetime(df_labels['start'], format='%Y:%m:%d %H:%M:%S')
    df_labels['finish'] =  pd.to_datetime(df_labels['finish'], format='%Y:%m:%d %H:%M:%S')
    # quick check to make sure all subjects are the same - only 1 sub per csv
    if (not (df_labels['sub'].eq(df_labels['sub'].iloc[0]).all())):
        print('Warning: Multiple subjects detected in csv, unusual for e4 data.')
    df['label']='Undefined' # add column with safe value for labels
    df['sub'] = np.NaN
    for index, row in df_labels.iterrows():
        #print(row['start'], row['finish'],row['label'])
        df.loc[row['start']:row['finish'],'label'] = row['label']
        df.loc[row['start']:row['finish'],'sub'] = row['sub']
    return df

def split_df_to_timeslice_nparrays(df, time_steps, step):
    """slice the df into segments of time_steps length and return X, y, sub
    ndarrays.  This is specific to an e4 ACC.csv file processed into dataframe.
    """
    N_FEATURES = 4 # maybe better to use # columns minus 2 (label and sub)
    # TODO - better yet pass in feature names and use length to set
    # if step = time_steps there is no overlap
    segments = []
    labels = []
    subject = []
    for i in range(0, len(df) - time_steps, step):
        df_segX = df[['accel_x', 'accel_y', 'accel_z','accel_ttl']].iloc[i: i + time_steps]
        df_lbl = df['label'].iloc[i: i + time_steps]
        df_sub = df['sub'].iloc[i: i + time_steps]
        # Save only if labels are the same for the entire segment and valid
        if (df_lbl.value_counts().iloc[0] != time_steps):
            #print('Segment at','contains multiple labels.  Discarding.')
            continue

        if 'Undefined' in df_lbl.values :
            #print('Segment contains Undefined labels.  Discarding')
            continue
        # Save only if sub is the same for the entire segment and valid
        if (df_sub.value_counts().iloc[0] != time_steps):
            print('Segment at','contains multiple subjects.  Discarding.')
            continue
        segments.append(df_segX.to_numpy())
        labels.append(df['label'].iloc[i])
        subject.append(df['sub'].iloc[i])

    # Bring the segments into a better shape, convert to nparrays
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)
    subject = np.asarray(subject)
    # both labels and sub are row arrays, change to single column arrays
    labels = labels[np.newaxis].T
    subject = subject[np.newaxis].T
    # check for nan - issue with resampled data
    bad_data_locations = np.argwhere(np.isnan(reshaped_segments))
    np.unique(bad_data_locations[:,0]) #[:,0] accesses just 1st column
    if (bad_data_locations.size==0):
        print("No NaN entries found")
    else:
        print("Warning: Output arrays contain NaN entries")
        print("execute print(X[99]) # to view single sample")
    return reshaped_segments, labels, subject

"""# Main Function to generate ndarrays"""

def get_X_y_sub(
    working_dir='support/content', # this directory will be created inside colab
    
    # you probably need to change this path to your google drive mount
    # zip_dir = '/content/drive/MyDrive/Colab Notebooks/imics_lab_repositories/load_data_time_series_dev/HAR/e4_wristband_Nov2019/zip_datafiles/sub1',
    zip_dir = 'support/zip_datafiles/sub1',

    zip_flist = [],
    # note the longer walk x25540_ zip file has not been labeled, this is for experiment only
    #zip_flist = ['1574625540_A01F11.zip'] # Old main to Alkek and back
    time_steps = 96,
    step = 32 #if equal to time_steps there will be no overlap of sliding window
):
    """processes e4 zip file and associated label csv file into X (data),
     y (labels), and sub (subject number) ndarrays.
     Returns X, y, sub, xys_info (a text file)
    """
    # create blank ndarrays to append to
    
    my_X = np.zeros(shape=(1,time_steps,4))
    my_y = np.full(shape=(1,1), fill_value='n/a',dtype='<U10') # unicode 10 char
    my_sub = np.zeros(shape=(1,1),dtype=int) # one subject number per entry
    for i in zip_flist:
        zip_ffname = zip_dir + '/' + i
        print('Processing ', i)
        unzip_e4_file(zip_ffname, working_dir)
        # following portion of code is unique to ACC only files
        ffname = working_dir + '/ACC.csv'
        col_labels = ['accel_x', 'accel_y', 'accel_z']
        my_df = df_from_e4_csv(ffname, col_labels)
        my_df = process_e4_accel(my_df)
        #print(my_df.head())
        print('Tag info (button presses) from tags.csv')
        tag_ffname = working_dir + '/tags.csv'
        show_tag_time(tag_ffname)
        # Generate associated csv filename, forces the long numbered filenames to match
        labels_ffname = os.path.splitext(zip_ffname)[0] + '_labels.csv'
        # print ('label file ', labels_ffname)
        my_df = label_df_from_csv (my_df, labels_ffname)
        my_df['label'].value_counts()
        print ("Label Counts\n",my_df['label'].value_counts())
        temp_X, temp_y, temp_sub = split_df_to_timeslice_nparrays(my_df, time_steps, step)
        #print(get_shapes([temp_X, temp_y, temp_sub]))
        #print(temp_X[:5]) # "head" for ndarray
        #print(temp_y[:5])
        #print(temp_sub[:5])
        my_X = np.vstack([my_X, temp_X])
        my_y = np.vstack([my_y, temp_y])
        my_sub = np.vstack([my_sub, temp_sub])
        # next line fails on windows, trying shutil to see if more portable
        #!rm -rf $working_dir # remove csv to start processing next file
        shutil.rmtree(working_dir)

    #delete first row placeholders
    X = np.delete(my_X, (0), axis=0) 
    y = np.delete(my_y, (0), axis=0) 
    sub = np.delete(my_sub, (0), axis=0)
    sub = sub.astype(int) # convert from float to int
    #print(get_shapes([X, y, sub]))
    # Print final counts for label ndarray - not quite as easy as pandas df
    unique, counts = np.unique(y, return_counts=True)
    print("Final Label Counts")
    print (np.asarray((unique, counts)).T)

    xys_info = 'e4 November 2019 zip files\n'
    xys_info += ' '.join([str(elem) for elem in zip_flist]) # conv list to string
    xys_info += '\nTime steps =' + str(time_steps) + ', Step =' + str(step) + ', no resample\n'
    xys_info += 'Final Shapes\n'
    #xys_info += get_shapes([X, y, sub])
    #print (xys_info)
    return X, y, sub, xys_info

def e4_load_dataset(
    verbose = True,
    incl_xyz_accel = False, # include component accel_x/y/z in ____X data
    incl_rms_accel = True, # add rms value (total accel) of accel_x/y/z in ____X data
    incl_val_group = False, # split train into train and validate
    split_subj = dict
                (train_subj = [11],
                validation_subj = [12],
                test_subj = [13]),
    one_hot_encode = True # make y into multi-column one-hot, one for each activity
    ):
    """calls e4_get_X_y_sub and processes the returned arrays by separating
    into _train, _validate, and _test arrays for X and y based on split_sub
    dictionary.  Note current dataset is single subject labeled as 11, 12, 13
    in order to exercise the code"""
    e4_flist = ['1574621345_A01F11.zip','1574622389_A01F11.zip', '1574624998_A01F11.zip']
    X, y, sub, xys_info = get_X_y_sub(zip_flist = e4_flist)
    log_info = 'Processing e4 files'+str(e4_flist)
    #remove component accel if needed
    if (not incl_xyz_accel):
        print("Removing component accel")
        X = np.delete(X, [0,1,2], 2)
    if (not incl_rms_accel):
        print("Removing total accel")
        X = np.delete(X, [3], 2)  
    #One-Hot-Encode y...there must be a better way when starting with strings
    #https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/

    if (one_hot_encode):
        # integer encode
        y_vector = np.ravel(y) #encoder won't take column vector
        le = LabelEncoder()
        integer_encoded = le.fit_transform(y_vector) #convert from string to int
        name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print("One-hot-encoding: category names -> int -> one-hot")
        print(name_mapping) # seems risky as interim step before one-hot
        log_info += "One Hot:" + str(name_mapping) +"\n\n"
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        print("One-hot-encoding",onehot_encoder.categories_)
        y=onehot_encoded
        #return X,y
    # split by subject number pass in dictionary
    sub_num = np.ravel(sub[ : , 0] ) # convert shape to (1047,)
    if (not incl_val_group):
        train_index = np.nonzero(np.isin(sub_num, split_subj['train_subj'] + 
                                        split_subj['validation_subj']))
        x_train = X[train_index]
        y_train = y[train_index]
    else:
        train_index = np.nonzero(np.isin(sub_num, split_subj['train_subj']))
        x_train = X[train_index]
        y_train = y[train_index]

        validation_index = np.nonzero(np.isin(sub_num, split_subj['validation_subj']))
        x_validation = X[validation_index]
        y_validation = y[validation_index]

    test_index = np.nonzero(np.isin(sub_num, split_subj['test_subj']))
    x_test = X[test_index]
    y_test = y[test_index]
    if (incl_val_group):
        return x_train, y_train, x_validation, y_validation, x_test, y_test
    else:
        return x_train, y_train, x_test, y_test
    
def e4_load_dataset_torch(args):
    """
    returns X -> (1047, 1, 96)
            y -> (1047) in 6 classes
    """
    x_train, y_train, x_test, y_test = e4_load_dataset(
        incl_xyz_accel=False,
        incl_rms_accel=True,
        one_hot_encode = True
    )
    x_train = torch.from_numpy(x_train).permute((0,2,1)).float()
    y_train = torch.from_numpy(y_train).long()
    x_test = torch.from_numpy(x_test).permute((0,2,1)).float()
    y_test = torch.from_numpy(y_test).long()
    
    return torch.cat((x_train, x_test)), torch.argmax(torch.cat((y_train, y_test)), dim=-1)


if __name__ == '__main__':
    x, y = e4_load_dataset_torch(
        # incl_xyz_accel=False,
        # incl_rms_accel=True
    )
    print(type(x))
    print(len(x))
            
    