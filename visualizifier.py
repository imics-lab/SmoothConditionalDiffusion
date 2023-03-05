#########################################
# Author:       Gentry Atkinson         #
# Organization: Texas State University  #
# Date:         04 Mar, 2023            #
#########################################

import argparse
import torch
import umap
import os

import numpy as np

from matplotlib import pyplot as plt
from get_fid_encoder import get_pretrained_encoder, get_fid_from_features
from datasets import load_dataset

dif_list = ['unconditional', 'conditional', 'soft_conditional']
ds_list = ['synthetic_5', 'unimib', 'mitbih', 'twristar']

def get_w2v_features(X : torch.Tensor):
    wave2vec = get_pretrained_encoder(args)
    f = [wave2vec( X[i,:,:], return_tensors='np')['input_values'] for i in range(X.shape[0])]
    f = torch.Tensor(f).squeeze()
    return f

def plot_umap_of_orig_vs_synth(
    args: any,
    X_orig : torch.Tensor,
    X_gen: torch.Tensor,
    f_name: str,
    extract_features=True
):
    SIZE = 2
    if extract_features:
        f_orig = get_w2v_features(X_orig)
        f_gen =  get_w2v_features(X_gen)
    else:
        f_orig = X_orig
        f_gen = X_gen
    reducer = umap.UMAP(n_neighbors=15, n_components=2)
    reducer.fit(f_orig)
    t_orig = reducer.transform(f_orig)
    t_gen = reducer.transform(f_gen)
    plt.figure()
    plt.scatter(t_orig[:,0], t_orig[:,1], c='blue',s=SIZE, marker='.', label="Original")
    plt.scatter(t_gen[:,0], t_gen[:,1], c='maroon',s=SIZE, marker='.', label="Generated")
    plt.axes = 'off'
    plt.savefig(f_name)



if __name__ == '__main__':
    if not os.path.isdir('plots'):
        os.mkdir('plots')

    # dif = "soft_conditional"
    # ds = "synthetic_5"

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.sampling_rate = 100
    args.data_path = 'data'
    
    fid_tab = {}

    for ds in ds_list:
        args.dataset = ds
        X_orig, _, y_orig, _ = load_dataset(args)
        f_orig = get_w2v_features(X_orig)
        fid_tab[ds] =  [-1]*len(dif_list)
        for i, dif in enumerate(dif_list):
            print("Plotting: ", ds, ' ', dif)
            args.mislab_rate = 0
            X_gen = torch.load(os.path.join('results', ds+"_"+dif+"_X_generated.pt"))
            y_gen = torch.load(os.path.join('results', ds+"_"+dif+"_y_generated.pt"))
            f_gen = get_w2v_features(X_gen)

            fname = ds + '_' + dif + ".pdf"

            plot_umap_of_orig_vs_synth(args, f_orig, f_gen, os.path.join('plots', fname), extract_features=False)
            fid_tab[ds][i] = get_fid_from_features(f_orig, f_gen)

    print(fid_tab)