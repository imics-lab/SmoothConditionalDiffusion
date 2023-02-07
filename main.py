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
from test_classifier import TestClassifier
import math

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

CUDA_DEV_NUM = ':5'

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="The dataset to run experiments on.", default='synthetic_5')
    parser.add_argument('--mislab_rate', help="Percentage of label noise to add.", default=0.05)
    parser.add_argument('--diffusion_model', help="A denoising model for reverse diffusion", default="UNet1d")
    parser.add_argument('--diffusion_style', help="unconditional, conditional, or probabilistic_conditional", default='unconditional')
    #parser.add_argument('--new_instances', help="The number of new instances of data to add", default=1000)
    parser.add_argument('--data_path', help="Directory for storing datasets", default='data')
    parser.add_argument('--run_path', help="Directory for storing runs outpus", default='runs')
    parser.add_argument('--data_cardinality', help="Dimensionality of data being processed", default='1d')
    parser.add_argument('--batch_size', help="Instance to train on per iteration", default=32)
    parser.add_argument('--lr', help="Learning Rate", default=0.001)
    parser.add_argument('--epochs', help="Number of epochs for training", default=1)
    parser.add_argument('--training_samples', help="number of samples to generate for each training epoch", default=10)
    parser.add_argument('--test_split', help="Portion of train data to hole out for test", default=0.2)
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

    #Prep Transition Matrix if needed
    T = None
    if args.diffusion_style == 'probabilistic_conditional':
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
        y_clean = torch.nn.functional.one_hot(y_clean.long()).long()

    print('Transition matrix: ', T)

    X_generated = None
    y_generated = None
    #Split dataset to train unconditional diffusion
    if args.diffusion_style=='unconditional':
        for i in range(args.num_classes):
            idxs = torch.where(y_noisy==i)[0]
            dataset = torch.utils.data.TensorDataset(X_original[idxs], torch.full((len(idxs),), i))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

            model, generator = load_diffuser(args)
            model, generator = train_diffusion(args, model, generator, dataloader, logger)
            #generator_seed = torch.randn_like(X_original)
            if X_generated == None:
                y_generated = torch.full((len(idxs),), i)
                X_generated = generator.sample(batch_size=len(idxs))
            else:
                y_generated = torch.concat([y_generated, torch.full((len(idxs),), i)])
                X_generated = torch.concat([X_generated, generator.sample(batch_size=len(idxs))])
    #Train with labels for both conditional approaches
    else:
        dataset = torch.utils.data.TensorDataset(X_original, y_noisy)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

        model, generator = load_diffuser(args)
        model, generator = train_diffusion(args, model, generator, dataloader, logger)
        #generator_seed = torch.randn_like(X_original)
        y_generated = torch.randint_like(y_clean, args.num_classes).to(args.device)
        X_generated = generator.sample(classes=y_generated)
    print('Shape of new data: ', X_generated.shape)
    #X_generated = X_generated.to(args.device)
    #y_generated = y_generated.to(args.device)

    #Train and test a classifier on JUST original data
    test_clsfr = TestClassifier(args)
    dataset = torch.utils.data.TensorDataset(X_original, y_noisy)
    test_size = math.ceil(args.test_split* len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_clsfr.train(args, dataloader, logger)

    #Train and test a classifier on JUST synthetic data
    test_clsfr = TestClassifier(args)
    dataset = torch.utils.data.TensorDataset(X_generated, y_generated)
    test_size = math.floor(args.test_split* len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    #Train and test a classifier on COMBINED data
    test_clsfr = TestClassifier(args)
    dataset = torch.utils.data.TensorDataset(torch.concat([X_original, X_generated]), torch.concat([y_noisy, y_generated]))
    test_size = math.floor(args.test_split* len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    