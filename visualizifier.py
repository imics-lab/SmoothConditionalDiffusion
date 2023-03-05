#########################################
# Author:       Gentry Atkinson         #
# Organization: Texas State University  #
# Date:         04 Mar, 2023            #
#########################################

import argparse
import pandas
import random
import torch
import umap
import os

import numpy as np

from matplotlib import pyplot as plt
from get_fid_encoder import get_pretrained_encoder, get_fid_from_features
from datasets import load_dataset

dif_list = ['unconditional', 'conditional', 'soft_conditional']
ds_list = ['synthetic_5', 'unimib', 'mitbih', 'twristar']

def get_w2v_features(args: any, X : torch.Tensor):
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
        f_orig = get_w2v_features(args, X_orig)
        f_gen =  get_w2v_features(args, X_gen)
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
    plt.axis('off')
    plt.savefig(f_name)

def prepare_fid_table(fids : dict, fname : str) -> None:
    df_dict = {
        'Dataset' : [],
        'Unconditional' : [],
        'Conditional' : [],
        'S. Conditional' : [],
        'Delta Unc. to Cond' : [],
        'Delta Unc. to S.Cond' : [],
    }
    for ds in fids.keys():
        df_dict['Dataset'].append(ds)
        df_dict['Unconditional'].append(fids[ds][0])
        df_dict['Conditional'].append(fids[ds][1])
        df_dict['S. Conditional'].append(fids[ds][2])
        del_un_to_con = (fids[ds][1] - fids[ds][0])/fids[ds][0]
        del_un_to_scon = (fids[ds][2] - fids[ds][0])/fids[ds][0]
        df_dict['Delta Unc. to Cond'].append(del_un_to_con)
        df_dict['Delta Unc. to S.Cond'].append(del_un_to_scon)

    df = pandas.DataFrame.from_dict(df_dict)
    print(df)
    with open(fname, 'w') as f:
        f.write(df.to_latex())

def plot_samples_from_data(
        args : any,
        X_orig : torch.Tensor, 
        y_orig : torch.Tensor,
        X_gen : torch.Tensor,
        y_gen : torch.Tensor,
        label : int,
        fname : str
):
    if y_gen.ndim > 1:
        y_gen = torch.argmax(y_gen, dim=-1)
    orig_instance = random.randint(0, len(X_orig))
    while y_orig[orig_instance] != label:
        orig_instance = random.randint(0, len(X_orig))

    gen_instance = random.randint(0, len(X_gen))
    while y_gen[gen_instance] != label:
        gen_instance = random.randint(0, len(X_gen))

    plt.figure()
    plt.plot(range(X_orig.shape[2]), X_orig[orig_instance, 0, :], c='blue')
    plt.plot(range(X_gen.shape[2]), X_gen[gen_instance, 0, :], c='maroon')
    plt.savefig(fname)

if __name__ == '__main__':
    if not os.path.isdir('plots'):
        os.mkdir('plots')

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.sampling_rate = 100
    args.data_path = 'data'
    
    fid_tab = {}
    label_to_print = 1

    for ds in ds_list:
        args.dataset = ds
        X_orig, _, y_orig, _ = load_dataset(args)
        f_orig = get_w2v_features(args, X_orig)
        fid_tab[ds] =  [-1]*len(dif_list)
        for i, dif in enumerate(dif_list):
            print("Plotting: ", ds, ' ', dif)
            args.mislab_rate = 0
            X_gen = torch.load(os.path.join('results', ds+"_"+dif+"_X_generated.pt"))
            y_gen = torch.load(os.path.join('results', ds+"_"+dif+"_y_generated.pt"))
            f_gen = get_w2v_features(args, X_gen)

            fname = ds + '_' + dif + ".pdf"

            plot_umap_of_orig_vs_synth(args, f_orig, f_gen, os.path.join('plots', fname), extract_features=False)
            fid_tab[ds][i] = get_fid_from_features(f_orig, f_gen)
            samp_name = os.path.join('plots', 'sample_'+ ds + '_' + dif + '_label_' + str(label_to_print) + '.pdf')
            plot_samples_from_data(args, X_orig, y_orig, X_gen, y_gen, label_to_print, samp_name)

    tab_name = os.path.join('plots', 'fid_table.txt')
    prepare_fid_table(fid_tab, tab_name)