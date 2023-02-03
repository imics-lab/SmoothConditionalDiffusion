#########################################
# Author:       Gentry Atkinson         #
# Organization: Texas State University  #
# Date:         03 Feb, 2023            #
#########################################

import torch
import os
from gen_ts_data import generate_signal_as_tensor

DATA_DIRECTORY = 'data'

def get_noisy_synthetic_dataset(args):
    SIGNAL_LENGTH = 128
    SET_LENGTH = 5001
    NUM_CHANNELS = 1
    sig_dict = {
        'avg_pattern_length' : [torch.randint(1, 10) for _ in args.num_classes],
        'avg_amplitude' : [torch.randint(1, 10) for _ in args.num_classes],
        'default_variance' : [torch.randint(1, 5) for _ in args.num_classes],
        'variance_pattern_length' : [torch.randint(1, 20) for _ in args.num_classes],
        'variance_amplitude' : [torch.randint(1, 5) for _ in args.num_classes],
    }
    y_clean = torch.randint(SET_LENGTH)
    y_noisy = torch.zeros(SET_LENGTH)
    X = torch.empty(SET_LENGTH, NUM_CHANNELS, SIGNAL_LENGTH)
    for i in range(SET_LENGTH):
        pass


def load_dataset(args):
    if not os.path.exists(DATA_DIRECTORY):
        os.mkdir(DATA_DIRECTORY)
    if args.dataset=='synthetic_5':
        if os.path.exists(os.path.join(DATA_DIRECTORY, 'synthetic_5.npy')):
            print("Synthetic 5 dataset located")
        else:
            print("Generating Synthetic 5 dataset")
            X, y_clean, y_noisy = get_noisy_synthetic_dataset(args)
