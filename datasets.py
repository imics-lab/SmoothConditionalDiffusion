#########################################
# Author:       Gentry Atkinson         #
# Organization: Texas State University  #
# Date:         03 Feb, 2023            #
#########################################

import torch
import os
import random
from gen_ts_data import generate_signal_as_tensor

DATA_DIRECTORY = 'data'

def get_noisy_synthetic_dataset(args, num_classes):
    SIGNAL_LENGTH = 128
    SET_LENGTH = 5001
    NUM_CHANNELS = 1
    random.seed(1899)
    sig_dict = {
        'avg_pattern_length' : [random.randint(5, 15) for _ in range(num_classes)],
        'avg_amplitude' : [random.randint(1, 10) for _ in range(num_classes)],
        'default_variance' : [random.randint(1, 5) for _ in range(num_classes)],
        'variance_pattern_length' : [random.randint(1, 20) for _ in range(num_classes)],
        'variance_amplitude' : [random.randint(1, 5) for _ in range(num_classes)],
    }
    y_clean = torch.randint(high=num_classes-1, size=(SET_LENGTH,)).int()
    y_noisy = torch.zeros(SET_LENGTH).int()
    X = torch.empty(SET_LENGTH, NUM_CHANNELS, SIGNAL_LENGTH)
    for i in range(SET_LENGTH):
        X[i, 0, :] = generate_signal_as_tensor(
            SIGNAL_LENGTH,
            sig_dict['avg_pattern_length'][y_clean[i]],
            sig_dict['avg_amplitude'][y_clean[i]],
            sig_dict['default_variance'][y_clean[i]],
            sig_dict['variance_pattern_length'][y_clean[i]],
            sig_dict['variance_amplitude'][y_clean[i]],
        )
        if random.random() <= args.mislab_rate:
            y_noisy[i] = random.randint(0, num_classes)
            while y_noisy[i] == y_clean[i]: y_noisy[i] = random.randint(0, num_classes)
        else:
            y_noisy[i] = y_clean[i]
        

    return X, y_clean, y_noisy


def load_dataset(args) -> tuple([torch.Tensor, torch.Tensor, torch.Tensor]):
    if not os.path.exists(DATA_DIRECTORY):
        os.mkdir(DATA_DIRECTORY)
    if args.dataset=='synthetic_5':
        if os.path.exists(os.path.join(DATA_DIRECTORY, 'synthetic_5_X.pt')):
            print("Synthetic 5 dataset located")
            X = torch.load(os.path.join(DATA_DIRECTORY, 'synthetic_5_X.pt'))
            y_clean = torch.load(os.path.join(DATA_DIRECTORY, 'synthetic_5_y_clean.pt'))
            y_noisy = torch.load(os.path.join(DATA_DIRECTORY, 'synthetic_5_y_noisy.pt'))

        else:
            print("Generating Synthetic 5 dataset")
            X, y_clean, y_noisy = get_noisy_synthetic_dataset(args, 5)
            torch.save(X, os.path.join(DATA_DIRECTORY, 'synthetic_5_X.pt'))
            torch.save(y_clean, os.path.join(DATA_DIRECTORY, 'synthetic_5_y_clean.pt'))
            torch.save(y_noisy, os.path.join(DATA_DIRECTORY, 'synthetic_5_y_noisy.pt'))

    return X, y_clean, y_noisy

if __name__ == '__main__':
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset = 'synthetic_5'
    args.mislab_rate = 0.05
    X, y_clean, y_noisy = load_dataset(args)
    print('Number of mislabeled instances: ', np.count_nonzero(y_clean != y_noisy))
    print('Measured mislabeling rate: ', np.count_nonzero(y_clean != y_noisy)/len(y_clean))
    print('Intended mislabeling rate: ', args.mislab_rate)
