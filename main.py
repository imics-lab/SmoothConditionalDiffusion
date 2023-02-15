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
import json
import umap
from matplotlib import pyplot as plt
from datetime import datetime

torch.multiprocessing.set_sharing_strategy('file_system')

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="The dataset to run experiments on.", default='mini_synthetic')
    parser.add_argument('--mislab_rate', help="Percentage of label noise to add.", default=0.05)
    parser.add_argument('--diffusion_model', help="A denoising model for reverse diffusion", default="UNet1d")
    parser.add_argument('--diffusion_style', help="unconditional, conditional, or probabilistic_conditional", default='unconditional')
    #parser.add_argument('--new_instances', help="The number of new instances of data to add", default=1000)
    parser.add_argument('--data_path', help="Directory for storing datasets", default='data')
    parser.add_argument('--run_path', help="Directory for storing training samples", default='runs')
    parser.add_argument('--data_cardinality', help="Dimensionality of data being processed", default='1d')
    parser.add_argument('--batch_size', help="Instances to train on per iteration", default=64)
    parser.add_argument('--lr', help="Learning Rate", default=0.001)
    parser.add_argument('--epochs', help="Number of epochs for training", default=150)
    parser.add_argument('--training_samples', help="number of samples to generate for each training epoch", default=4)
    parser.add_argument('--test_split', help="Portion of train data to hole out for test", default=0.2)
    parser.add_argument('--dev_num', help="Device number for running experiments on GPU", default=4)
    parser.add_argument('--time_steps', help="Time steps for noising/denoising.", default=1000)
    args = parser.parse_args()
    return args

results_dic = {
    'Accuracy on original data' : 0,
    'Accuracy on synthetic data' : 0,
    'Accuracy on both' : 0,
    'FID' : 0,
    'Label distance' : 0,
    'Time' : None,
    'Time_steps' : None
}

if __name__ == '__main__':
    args = load_args()
    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)
    if not os.path.exists('results'):
        os.mkdir('results')
    logger = SummaryWriter(os.path.join("runs", args.run_path))
    if torch.cuda.is_available():
        args.device = 'cuda:' + str(args.dev_num)
    else:
        args.device = 'cpu'
    #args.device = 'cpu'
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
        y_clean = torch.nn.functional.one_hot(y_clean.long(), num_classes=args.num_classes).long()

    print('Transition matrix: ', T)
    args.seq_length = X_original.shape[2]

    X_generated = None
    y_generated = None
    #Split dataset to train unconditional diffusion
    if args.diffusion_style=='unconditional':
        for i in range(args.num_classes):
            logging.info(f"Generating Samples of Class {i}:")
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
        if args.diffusion_style=='conditional':
            y_generated = torch.randint_like(y_clean, args.num_classes)
        elif args.diffusion_style=='probabilistic_conditional':
            y_generated = torch.randint(args.num_classes, (len(y_clean),))
            y_generated = torch.mul(torch.nn.functional.one_hot(y_generated, num_classes=args.num_classes), args.num_classes-1)
        X_generated = generator.sample(classes=y_generated.to(args.device))
    print('Shape of new data: ', X_generated.shape)

    #Train and test a classifier on JUST original data
    test_clsfr = TestClassifier(args)   
    acc = test_clsfr.train_and_test_classifier(args, X_original, y_noisy, logger)
    results_dic['Accuracy on original data'] = acc

    #Train and test a classifier on JUST synthetic data
    test_clsfr = TestClassifier(args)  
    acc = test_clsfr.train_and_test_classifier(args, X_generated, y_generated, logger)
    results_dic['Accuracy on synthetic data'] = acc

    #Train and test a classifier on COMBINED data
    test_clsfr = TestClassifier(args)
    acc = test_clsfr.train_and_test_classifier(
            args, 
            X_original, 
            y_noisy,  
            logger,
            X_generated,
            y_generated
    )
    results_dic['Accuracy on both'] = acc

    #Save the results
    results_dic['Time'] = str(datetime.now())
    results_dic['Time_steps'] = args.time_steps
    print(results_dic)
    with open(f'results/{args.diffusion_style}_{args.diffusion_model}_{args.dataset}_accuracies.txt', 'w') as f:
        f.write(json.dumps(results_dic))

    #Print umaps of original vs. generated data
    f_original = get_features_for_set(np.array(torch.permute(X_original, (0, 2, 1)).cpu().detach()))
    f_synthetic = get_features_for_set(np.array(torch.permute(X_original, (0, 2, 1)).cpu().detach()))
    reducer = umap.UMAP(n_neighbors=15, n_components=2)
    embedding_orig = reducer.fit_transform(f_original)
    embedding_syn = reducer.fit_transform(f_synthetic)

    plt.figure()
    plt.scatter(embedding_orig[:,0], embedding_orig[:,1], c='blue', marker=',')
    plt.scatter(embedding_syn[:,0], embedding_syn[:,1], c='maroon', marker=',')
    plt.savefig(os.path.join('results', f'{args.dataset}_{args.diffusion_style}.pdf'))

    #Preserve the generated tensors
    torch.save(X_generated, os.path.join('results', f'{args.dataset}_{args.diffusion_style}_X_generated.pt'))
    torch.save(y_generated, os.path.join('results', f'{args.dataset}_{args.diffusion_style}_y_generated.pt'))
    torch.save(model.state_dict(), os.path.join('results', f'{args.dataset}_{args.diffusion_style}_denoiser.pt'))
    torch.save(generator.state_dict(), os.path.join('results', f'{args.dataset}_{args.diffusion_style}_generator.pt'))


    
    