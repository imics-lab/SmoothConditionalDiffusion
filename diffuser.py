#########################################
# Author:       Gentry Atkinson         #
# Organization: Texas State University  #
# Date:         03 Feb, 2023            #
#########################################

import torch
from torch import optim
from support.modules1D import Unet1D, GaussianDiffusion1D
import logging
import tqdm
import os
from support.utils import save_signals, save_checkpoint

def load_diffuser(args):
    model = None
    diffusion = None
    if args.diffusion_style == 'unconditional':
        if args.diffusion_model == 'UNet1d':
            assert args.data_cardinality == '1d', "Data cardinality must match denoising model"
            model = Unet1D(
                dim = 64,
                dim_mults = (1, 2, 4, 8),
                channels = 1
            ).to(args.device)
            # seq_length must be able to divided by dim_mults
            diffusion = GaussianDiffusion1D(
                model,
                seq_length = 128,
                timesteps = 1000,
                objective = 'pred_v'
            ).to(args.device)
        else:
            print(f"Denoising Model choice: {args.diffusion_model} is not supported")
    else:
        print(f"Diffusion Style choice: {args.diffusion_style} is not supported")
    return model, diffusion

def train_diffusion(args, model, diffusion, dataloader, logger):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (signals, _) in enumerate(pbar):
            signals = signals.to(args.device).to(torch.float)
            loss = diffusion(signals)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar("loss", loss.item(), global_step=epoch * l + i)

        sampled_signals = diffusion.sample(batch_size = 10)
        sampled_signals.shape # (10, 1, 128)
        
        is_best = False
        
        save_signals(sampled_signals, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, os.path.join("checkpoint", args.run_name))
    
    return model, diffusion