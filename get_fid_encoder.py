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

if __name__ == '__main__':
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args .data_path = 'data'
    args.dataset = 'unimib'
    args.mislab_rate = 0.05
    t = get_pretrained_encoder(args)
    X = torch.randn(size=(1, 128))
    f = t(X)
    print('X shape: ', X.shape)
    print(len(f['input_values']))