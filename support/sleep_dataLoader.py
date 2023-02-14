import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne.time_frequency import psd_welch

from torch.utils.data import Dataset, DataLoader
import torch

class SleepStage_Dataset(Dataset):
    def __init__(self, data_mode = 'Train'):
        
        self.data_mode = data_mode
        if self.data_mode == 'Train':
            self.subjects = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        else:
            self.subjects = [11, 12, 13]
        
        subject_files = []
        for subject in self.subjects:
            curr_subject_file = fetch_data(subjects=[subject], recording=[1])
            subject_files.append(curr_subject_file)
        #print(subject_files[0]) #[['/home/x_l30/mne_data/physionet-sleep-data/SC4001E0-PSG.edf', '/home/x_l30/mne_data/physionet-sleep-data/SC4001EC-Hypnogram.edf']]
        
        raw_train = []
        annot_train = []
        for subject_file in subject_files:
            raw = mne.io.read_raw_edf(subject_file[0][0], stim_channel='marker', misc=['rectal'])
            raw_train.append(raw)
            annot = mne.read_annotations(subject_file[0][1])
            annot_train.append(annot)
            
        print(len(raw_train))
        print(len(annot_train))
        
        annotation_desc_2_event_id = {'Sleep stage W': 1,
                              'Sleep stage 1': 2,
                              'Sleep stage 2': 3,
                              'Sleep stage 3': 4,
                              'Sleep stage 4': 4,
                              'Sleep stage R': 5}
        
        # create a new event_id that unifies stages 3 and 4
        event_id = {'Sleep stage W': 1,
                    'Sleep stage 1': 2,
                    'Sleep stage 2': 3,
                    'Sleep stage 3/4': 4,
                    'Sleep stage R': 5}
        
        data = []
        labels = []
        for raw, annot in zip(raw_train, annot_train):
            #raw.set_annotations(annot, emit_warning=False)
            
            # keep last 30-min wake events before sleep and first 30-min wake events after
            # sleep and redefine annotations on raw data
            annot.crop(annot[1]['onset'] - 30 * 60,
                             annot[-2]['onset'] + 30 * 60)
            raw.set_annotations(annot, emit_warning=False)

            events, _ = mne.events_from_annotations(
                raw, event_id=annotation_desc_2_event_id, chunk_duration=30.)

            #split to epochs
            tmax = 30. - 1. / raw.info['sfreq']  # tmax in included

            epochs = mne.Epochs(raw=raw, events=events,
                                event_id=event_id, tmin=0., tmax=tmax, baseline=None)
            epochs_data = epochs.get_data()
            epochs_labels = epochs.events[:, 2]
            
            data.extend(epochs_data)
            labels.extend(epochs_labels)
            
        data = np.array(data)
        labels = np.array(labels)
        
#         self.data = epochs_train.get_data()
        self.data = data.reshape(data.shape[0], data.shape[1], 1, data.shape[2])
#         self.labels = epochs_train.events[:, 2]
        self.labels = labels - 1
        print(self.data.shape)
        print(self.labels.shape)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]