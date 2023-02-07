#########################################
# Author:       Gentry Atkinson         #
# Organization: Texas State University  #
# Date:         03 Feb, 2023            #
#########################################

import argparse
import torch
from datasets import load_dataset, expand_labels
from diffuser import load_diffuser, train_diffusion
from torch.utils.tensorboard import SummaryWriter
import logging
import os
from multiprocessing import cpu_count
from hoc import get_T_global_min_new
from ts_feature_toolkit import get_features_for_set
import numpy as np

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

CUDA_DEV_NUM = ':3'

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="The dataset to run experiments on.", default='synthetic_5')
    parser.add_argument('--mislab_rate', help="Percentage of label noise to add.", default=0.05)
    parser.add_argument('--diffusion_model', help="A denoising model for reverse diffusion", default="UNet1d")
    parser.add_argument('--diffusion_style', help="unconditional, conditional, or probabalistic_conditional", default='probabalistic_conditional')
    parser.add_argument('--new_instances', help="The number of new instances of data to add", default=1000)
    parser.add_argument('--data_path', help="Directory for storing datasets", default='data')
    parser.add_argument('--run_path', help="Directory for storing runs outpus", default='runs')
    parser.add_argument('--data_cardinality', help="Dimensionality of data being processed", default='1d')
    parser.add_argument('--batch_size', help="Instance to train on per iteration", default=32)
    parser.add_argument('--lr', help="Learning Rate", default=0.001)
    parser.add_argument('--epochs', help="Number of epochs for training", default=200)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = load_args()
    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)
    logger = SummaryWriter(os.path.join("runs", args.run_path))
    if torch.cuda.is_available():
        args.device = 'cuda' + CUDA_DEV_NUM
    else:
        args.device = 'cpu'
    args.num_workers = cpu_count()
    print("---Experiments on Probilbalistic Conditional Diffusion---")

    X_original, y_clean, y_noisy, T = load_dataset(args)

    T = None
    if args.diffusion_style == 'probabalistic_conditional':
        print('Estimating Transition Matrix')
        #swap to channels-last nd array for tsfresh
        f = get_features_for_set(np.array(torch.permute(X_original, (0, 2, 1)).cpu().detach()))
        #no back to torch
        f = torch.from_numpy(f).to(args.device)
        ds = {
            'feature' : f,
            'noisy_label' : y_noisy
        }       
        T, P, global_dic = get_T_global_min_new(args, ds, T0=torch.eye(args.num_classes), all_point_cnt=args.cnt//5, global_dic={})
        T = torch.from_numpy(T).to(args.device)
        y_noisy = expand_labels(y_noisy, T)
        y_clean = torch.nn.functional.one_hot(y_clean).int()

    print('Transition matrix: ', T)
    dataset = torch.utils.data.TensorDataset(X_original, y_noisy)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model, generator = load_diffuser(args)
    model, generator = train_diffusion(args, model, generator, dataloader, logger)