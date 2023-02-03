#########################################
# Author:       Gentry Atkinson         #
# Organization: Texas State University  #
# Date:         03 Feb, 2023            #
#########################################

import torch
import os

DATA_DIRECTORY = 'data'

def load_dataset(args):
    if not os.path.exists(DATA_DIRECTORY):
        os.mkdir(DATA_DIRECTORY)
    if args.dataset=='synthetic_5':
        if os.path.exists(os.path.join(DATA_DIRECTORY, 'synthetic_5.npy')):
            print("Synthetic 5 dataset located")
        else:
            print("Generating Synthetic 5 dataset")
