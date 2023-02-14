#########################################
# Author:       Gentry Atkinson         #
# Organization: Texas State University  #
# Date:         03 Feb, 2023            #
#########################################

#Incoming datasets:
#   Apnea ECG- https://www.kaggle.com/datasets/ecerulm/apneaecg
#   PTB diagnostic ECG- https://www.kaggle.com/datasets/openmark/ptb-diagnostic-ecg-database
#   SWELL- https://www.kaggle.com/datasets/qiriro/swell-heart-rate-variability-hrv
#   Electromyography dataset- https://www.kaggle.com/datasets/nccvector/electromyography-emg-dataset
#   PSG apnea? Yeah, not a bad idea
#   HAR from Lee
#   Sleep EEG

import torch
import os
import random
from gen_ts_data import generate_signal_as_tensor
import zipfile
from support.MITBIH import mitbih_allClass
from support.har_dataloader import unimib_load_dataset

def get_noisy_synthetic_dataset(args, num_classes, set_length=5001, num_channels=1):
    SIGNAL_LENGTH = 128
    random.seed(1899)
    sig_dict = {
        'avg_pattern_length' : [random.randint(5, 15) for _ in range(num_classes)],
        'avg_amplitude' : [random.randint(1, 10) for _ in range(num_classes)],
        'default_variance' : [random.randint(1, 5) for _ in range(num_classes)],
        'variance_pattern_length' : [random.randint(1, 20) for _ in range(num_classes)],
        'variance_amplitude' : [random.randint(1, 5) for _ in range(num_classes)],
    }
    y_clean = torch.randint(high=num_classes, size=(set_length,)).int()
    y_noisy = torch.zeros(set_length).int()
    X = torch.empty(set_length, num_channels, SIGNAL_LENGTH)
    for i in range(set_length):
        X[i, 0, :] = generate_signal_as_tensor(
            SIGNAL_LENGTH,
            sig_dict['avg_pattern_length'][y_clean[i]],
            sig_dict['avg_amplitude'][y_clean[i]],
            sig_dict['default_variance'][y_clean[i]],
            sig_dict['variance_pattern_length'][y_clean[i]],
            sig_dict['variance_amplitude'][y_clean[i]],
        )
        if random.random() <= args.mislab_rate:
            y_noisy[i] = random.randint(0, num_classes-1)
            while y_noisy[i] == y_clean[i]: y_noisy[i] = random.randint(0, num_classes-1)
        else:
            y_noisy[i] = y_clean[i]
        
    y_clean = y_clean.long()
    y_noisy = y_noisy.long()
    return X, y_clean, y_noisy

def download_mit_bih_dataset(args):
    #assert os.path.exists('~/kaggle.json/kaggle.json'), "A Kaggle API token is required"
    os.system('kaggle datasets download shayanfazeli/heartbeat')
    os.system(f'mv heartbeat.zip {args.data_path}/.')
    with zipfile.ZipFile(f'{args.data_path}/heartbeat.zip', 'r') as zip_ref:
        zip_ref.extractall(args.data_path)
    os.system(f'rm {args.data_path}/heartbeat.zip')

def get_noisy_labels_from_clean(args, y_clean):
    y_noisy = torch.zeros_like(y_clean)
    for i, y in enumerate(y_clean):
        if random.random() <= args.mislab_rate:
            y_noisy[i] = random.randint(0, args.num_classes-1)
            while y_noisy[i] == y_clean[i]: y_noisy[i] = random.randint(0, args.num_classes-1)
        else:
            y_noisy[i] = y_clean[i]
    return y_noisy

def expand_labels(y, T):
    num_classes = len(T)
    y_expanded = torch.empty((len(y), num_classes))
    for i, yi in enumerate(y):
        y_expanded[i] = T[yi]
    #y_expanded = y_expanded*100
    y_expanded = torch.argsort(y_expanded, descending=True)
    return y_expanded.long()

def get_unimib(args):
    uds = unimib_load_dataset(
        incl_xyz_accel=False,
        incl_rms_accel=True,
        single_class = False
    )
    X, y = uds[:]
    # X = torch.squeeze(torch.from_numpy(X))
    X = torch.squeeze(torch.from_numpy(X), dim=2)
    y = torch.argmax(torch.from_numpy(y), dim=-1).long()
    return X, y

def load_dataset(args) -> tuple([torch.Tensor, torch.Tensor, torch.Tensor]):
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    args.num_classes = 0
    X = None
    y_clean = None
    y_noisy = None
    T = None
    if args.dataset=='synthetic_5':
        args.num_classes = 5
        if os.path.exists(os.path.join(args.data_path, 'synthetic_5_X.pt')):
            print("Synthetic 5 dataset located")
            X = torch.load(os.path.join(args.data_path, 'synthetic_5_X.pt'))
            y_clean = torch.load(os.path.join(args.data_path, 'synthetic_5_y_clean.pt')).int()
            y_noisy = torch.load(os.path.join(args.data_path, 'synthetic_5_y_noisy.pt')).int()

        else:
            print("Generating Synthetic 5 dataset")
            X, y_clean, y_noisy = get_noisy_synthetic_dataset(args, 5)
            torch.save(X, os.path.join(args.data_path, 'synthetic_5_X.pt'))
            torch.save(y_clean, os.path.join(args.data_path, 'synthetic_5_y_clean.pt'))
            torch.save(y_noisy, os.path.join(args.data_path, 'synthetic_5_y_noisy.pt'))
    elif args.dataset=='mitbih':
        args.num_classes = 5
        if os.path.exists(os.path.join(args.data_path, 'mitbih_train.csv')):
            print("Found MIT Arythmia Dataset")
        else:
            print("Downloading MIT Arythmia Dataset")
            download_mit_bih_dataset(args)
        filename = os.path.join(args.data_path, 'mitbih_train.csv')
        data = mitbih_allClass(isBalanced = True, filename=filename, n_samples=2000)
        X, y_clean = data[:]
        X = torch.Tensor(X)
        y_clean = torch.Tensor(y_clean).int()
        print('MIT BIH X shape: ', X.shape)
        print('MIT BIH y_clean shape: ', y_clean.shape)
        y_noisy = get_noisy_labels_from_clean(args, y_clean)
        y_noisy = y_noisy.int()
    elif args.dataset == 'mini_synthetic':
        X, y_clean, y_noisy = get_noisy_synthetic_dataset(args, 2, 101)
        args.num_classes = 2
    elif args.dataset == 'unimib':
        args.num_classes = 9
        X, y_clean = get_unimib(args)
        y_noisy = get_noisy_labels_from_clean(args, y_clean)
    else:
        print(f'Chosen dataset: {args.dataset} is not supported')

    # print('Max y_clean: ', torch.max(y_clean))
    # print('Max y_noisy: ', torch.max(y_noisy))
    
    args.cnt = len(X)
    return X, y_clean, y_noisy, T

if __name__ == '__main__':
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args .data_path = 'data'
    args.dataset = 'unimib'
    args.mislab_rate = 0.05
    X, y_clean, y_noisy, _ = load_dataset(args)
    print('X shape: ', X.shape)
    print('y_clean shape: ', y_clean.shape)
    print('y_noisy shape: ', y_noisy.shape)
    print('Number of mislabeled instances: ', np.count_nonzero(y_clean != y_noisy))
    print('Measured mislabeling rate: ', np.count_nonzero(y_clean != y_noisy)/len(y_clean))
    print('Intended mislabeling rate: ', args.mislab_rate)
