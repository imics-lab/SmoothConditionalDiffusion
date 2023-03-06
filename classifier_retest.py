#########################################
# Author:       Gentry Atkinson         #
# Organization: Texas State University  #
# Date:         04 Mar, 2023            #
#########################################

import os
import torch
import pandas
import argparse
import numpy as np
from sklearn.svm import SVC
from datasets import load_dataset
from visualizifier import get_w2v_features
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


dif_list = ['unconditional', 'conditional', 'soft_conditional']
ds_list = ['synthetic_5', 'unimib', 'mitbih', 'twristar']

if __name__ == '__main__':
    if not os.path.isdir('plots'):
        os.mkdir('plots')

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.sampling_rate = 100
    args.data_path = 'data'
    args.mislab_rate = 0.05
    
    svm_tab = {
        'Dataset' : [],
        'Diffusion' : [],
        'Orig Acc' : [],
        'Syn Acc' : [],
        'Combined Acc' : []
    }
    TEST_SIZE = 0.2

    for ds in ds_list:
        args.dataset = ds
        X_orig, _, y_orig, _ = load_dataset(args)
        y_orig = y_orig.long().numpy()
        f_orig = get_w2v_features(args, X_orig)
        f_orig = f_orig.float().numpy()
        train_X_1, test_X_1, train_y_1, test_y_1 = train_test_split(f_orig, y_orig, shuffle=True, test_size=TEST_SIZE)
        for i, dif in enumerate(dif_list):
            print("Classifying: ", ds, ' ', dif)
            X_gen = torch.load(os.path.join('results', ds+"_"+dif+"_X_generated.pt"))
            y_gen = torch.load(os.path.join('results', ds+"_"+dif+"_y_generated.pt"))
            if y_gen.ndim>1:
                y_gen = torch.argmax(y_gen, dim=-1)
            f_gen = get_w2v_features(args, X_gen)
            y_gen = y_gen.long().numpy()
            f_gen = f_gen.float().numpy()
            
            
            svm_tab['Dataset'].append(ds)
            svm_tab['Diffusion'].append(dif)
            
            train_X_2, test_X_2, train_y_2, test_y_2 = train_test_split(f_gen, y_gen, shuffle=True, test_size=TEST_SIZE)
            train_X_3 = np.concatenate((train_X_1, train_X_2), axis=0)
            train_y_3 = np.concatenate((train_y_1, train_y_2), axis=0)
            test_X_3 = np.copy(test_X_1)
            test_y_3 = np.copy(test_y_1)

            clsf_1 = SVC(kernel='poly')
            clsf_1.fit(train_X_1, train_y_1)
            pred_1 = clsf_1.predict(test_X_1)
            svm_tab['Orig Acc'].append(accuracy_score(test_y_1, pred_1))

            clsf_2 = SVC(kernel='poly')
            clsf_2.fit(train_X_2, train_y_2)
            pred_2 = clsf_2.predict(test_X_2)
            svm_tab['Syn Acc'].append(accuracy_score(test_y_2, pred_2))

            clsf_3 = SVC(kernel='poly')
            clsf_3.fit(train_X_3, train_y_3)
            pred_3 = clsf_3.predict(test_X_3)
            svm_tab['Combined Acc'].append(accuracy_score(test_y_3, pred_3))

    df = pandas.DataFrame.from_dict(svm_tab)
    print(df)
    with open(os.path.join('plots', 'svm_acc.txt'), 'w+') as f:
        f.write(df.to_string())
