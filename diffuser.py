#########################################
# Author:       Gentry Atkinson         #
# Organization: Texas State University  #
# Date:         03 Feb, 2023            #
#########################################

import torch
from torch import optim
from support.modules1D import Unet1D, GaussianDiffusion1D
from support.modules1D_cls_free import Unet1D_cls_free, GaussianDiffusion1D_cls_free
import logging
from tqdm import tqdm
import os
from support.utils import save_signals, save_checkpoint, save_signals_cls_free

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
    if args.diffusion_style == 'conditional':
        if args.diffusion_model == 'UNet1d':
            assert args.data_cardinality == '1d', "Data cardinality must match denoising model"
            model = Unet1D_cls_free(
                dim = 64,
                dim_mults = (1, 2, 4, 8),
                num_classes = args.num_classes,
                cond_drop_prob = 0.5,
                channels = 1
            ).to(args.device)

        
            diffusion = GaussianDiffusion1D_cls_free(
                model,
                seq_length = 128,
                timesteps = 1000
            ).to(args.device)
    else:
        print(f"Diffusion Style choice: {args.diffusion_style} is not supported")
    return model, diffusion

def train_unconditional(args, model, diffusion, dataloader, logger, optimizer):
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
        
        save_signals(sampled_signals, os.path.join(args.run_path, f"{args.dataset}_{args.diffusion_style}_training_{epoch}.jpg"))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, args.run_path)
    return model, diffusion

def train_conditional(args, model, diffusion, dataloader, logger, optimizer):
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (signals, labels) in enumerate(pbar):
            signals = signals.to(args.device).to(torch.float)
            labels = labels.to(args.device).to(torch.long)
            loss = diffusion(signals, classes = labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logger.add_scalar("loss", loss.item(), global_step=epoch * l + i)

        labels = torch.randint(0, args.num_classes, (10,)).to(args.device)
        sampled_signals = diffusion.sample(
            classes = labels,
            cond_scale = 3.)
        sampled_signals.shape # (10, 1, 128)
        
        is_best = False
        
        save_signals_cls_free(sampled_signals, labels, os.path.join(args.run_path, f"{args.dataset}_{args.diffusion_style}_training_{epoch}.jpg"))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, os.path.join("checkpoint", args.run_name))
    return model, diffusion

def train_diffusion(args, model, diffusion, dataloader, logger):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    l = len(dataloader)
    if args.diffusion_style == 'unconditional':
        model, diffusion = train_unconditional(args, model, diffusion, dataloader, logger, optimizer)
    elif args.diffusion_style == 'conditional':
        model, diffusion = train_conditional(args, model, diffusion, dataloader, logger, optimizer)
    else:
        print(f'Selected diffusion style: {args.diffusion_style} is not supported.')
        model, diffusion = None, None

    
    
    return model, diffusion