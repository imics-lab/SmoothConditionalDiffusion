#########################################
# Author:       Gentry Atkinson         #
# Organization: Texas State University  #
# Date:         03 Feb, 2023            #
#########################################

from transformers import Wav2Vec2FeatureExtractor
import torch


def get_pretrained_encoder(args):
    args.feat_extract_norm = 'layer'
    t = Wav2Vec2FeatureExtractor(args)
    return t

def get_fid_from_features(f_original, f_synthetic):
    mu_1 = torch.mean(f_original)
    mu_2 = torch.mean(f_synthetic)
    sig_1 = torch.std(f_original)
    sig_2 = torch.std(f_synthetic)
    # return abs(mu_1 - mu_2)**2 + torch.trace(sig_1 + sig_2 - 2*torch.sqrt(torch.sqrt(sig_1)*sig_2*torch.sqrt(sig_1)))
    return (abs(mu_1 - mu_2)**2 + (sig_1 + sig_2 - 2*torch.sqrt(torch.sqrt(sig_1)*sig_2*torch.sqrt(sig_1)))).item()



if __name__ == '__main__':
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args .data_path = 'data'
    args.dataset = 'unimib'
    args.mislab_rate = 0.05
    t = get_pretrained_encoder(args)
    X_1 = torch.randn(size=(1, 128))
    f_1 = t(X_1)
    X_2 = torch.randn(size=(1, 128))
    f_2 = t(X_2)
    f_1 = torch.Tensor(f_1['input_values'])
    f_2 = torch.Tensor(f_2['input_values'])
    print('X_1 shape: ', X_1.shape)
    print('X_2 shape: ', X_2.shape)
    print('Dist: ', get_fid_from_features(f_1, f_2))
    