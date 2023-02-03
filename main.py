#########################################
# Author:       Gentry Atkinson         #
# Organization: Texas State University  #
# Date:         03 Feb, 2023            #
#########################################

import argparse
import torch
from datasets import load_dataset
from diffuser import load_diffuser

CUDA_DEV_NUM = ':0'

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="The dataset to run experiments on.", default='synthetic_5')
    parser.add_argument('--mislab_rate', help="Percentage of label noise to add.", default='0.05')
    parser.add_argument('--diffusion_model', help="A denoising model for reverse diffusion", default="UNet1d")
    parser.add_argument('--difusion_style', help="Unconditional, conditional, or probabalistic_conditional", default='probabalistic_conditional')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = load_args()
    if torch.cuda.is_available():
        args.device = 'cuda' + CUDA_DEV_NUM
    else:
        args.device = 'cpu'
    print("---Experiments on Probilbalistic Conditional Diffusion---")

    X_original, y_clean, y_noisy = load_dataset(args)
    generator = load_diffuser(args)