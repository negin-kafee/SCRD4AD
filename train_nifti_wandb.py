#!/usr/bin/env python3
"""
SCRD4AD Training Script with NIfTI Support and W&B Logging
Loads NIfTI volumes directly instead of pre-processed PNG slices.
"""

import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from models.rd4ad_mlp import RdadAtten
from dataset_nifti import NiftiSliceDataset, NiftiSliceDatasetCached
import torch.backends.cudnn as cudnn
import argparse
from utils import global_cosine_param

from torch.nn import functional as F

import warnings
import logging

warnings.filterwarnings("ignore")

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args, print_fn, device):
    setup_seed(111)

    total_iters = args.total_iters
    batch_size = args.batch_size

    # Use NIfTI dataset
    print_fn(f"Loading NIfTI data from: {args.nifti_dirs}")
    
    if args.cached:
        print_fn("Using cached (in-memory) dataset")
        train_data = NiftiSliceDatasetCached(
            args.nifti_dirs, 
            min_slice_pct=args.min_slice_pct,
            max_slice_pct=args.max_slice_pct,
            mylambda=args.noise_lambda
        )
    else:
        print_fn("Using on-the-fly dataset loading")
        train_data = NiftiSliceDataset(
            args.nifti_dirs,
            min_slice_pct=args.min_slice_pct,
            max_slice_pct=args.max_slice_pct,
            mylambda=args.noise_lambda
        )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, 
        num_workers=args.num_workers, drop_last=False, pin_memory=True
    )
    
    model = RdadAtten()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    print_fn(f'Training samples: {len(train_data)}')
    print_fn(f'Model parameters: {count_parameters(model):,}')

    # Create experiment-specific model directory
    model_dir = os.path.join(args.output_dir, args.experiment_name, 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)

    it = 0
    model = model.to(device=device)
    
    epochs_needed = total_iters // len(train_dataloader) + 1
    print_fn(f'Training for {epochs_needed} epochs ({total_iters} iterations)')
    
    for epoch in range(epochs_needed):
        model.train()
        loss_list = []
        for img, img_noise in train_dataloader:
            img = img.to(device)
            img_noise = img_noise.to(device)
            inputs, outputs, param = model(img)
            inputs_noise, outputs_noise, _ = model(img_noise)
            
            loss1 = global_cosine_param(inputs, outputs, param)
            loss2 = global_cosine_param(inputs_noise, outputs, param)
            loss = loss1 / loss2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            
            if (it + 1) % 1000 == 0:
                avg_loss = np.mean(loss_list)
                print_fn(f'iter [{it}/{total_iters}], loss:{avg_loss:.4f}')
                loss_list = []
                
                # Log to W&B
                if WANDB_AVAILABLE and args.wandb_mode != 'disabled':
                    wandb.log({
                        'iteration': it + 1,
                        'loss': avg_loss,
                        'epoch': epoch,
                    })
                
                # Save checkpoint every 10k iterations
                if (it + 1) % 10000 == 0:
                    ckpt_path = os.path.join(model_dir, f'checkpoint_{it+1}.pth')
                    torch.save(model.state_dict(), ckpt_path)
                    print_fn(f'Saved checkpoint to {ckpt_path}')
            
            it += 1
            if it == total_iters:
                break
        
        if it >= total_iters:
            break

    # Final save
    final_path = os.path.join(model_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print_fn(f'Saved final model to {final_path}')
    
    # Log final metrics to W&B
    if WANDB_AVAILABLE and args.wandb_mode != 'disabled':
        wandb.log({
            'final_loss': np.mean(loss_list) if loss_list else avg_loss,
            'total_iterations': it,
        })
        wandb.finish()

    return avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SCRD4AD NIfTI Training with W&B')
    
    # Data arguments
    parser.add_argument('--nifti_dirs', nargs='+', required=True, 
                        help='Directories containing .nii.gz files')
    parser.add_argument('--min_slice_pct', type=int, default=10,
                        help='Minimum slice percentage (0-100)')
    parser.add_argument('--max_slice_pct', type=int, default=90,
                        help='Maximum slice percentage (0-100)')
    parser.add_argument('--cached', action='store_true',
                        help='Cache all slices in memory (faster but needs more RAM)')
    
    # Training arguments
    parser.add_argument('--experiment_name', type=str, required=True, 
                        help='Name of experiment')
    parser.add_argument('--output_dir', type=str, 
                        required=True,
                        help='Output directory')
    parser.add_argument('--total_iters', type=int, default=40000,
                        help='Total training iterations')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--noise_lambda', type=float, default=0.2,
                        help='Simplex noise intensity')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--gpu', default='0', type=str, 
                        help='GPU id to use')
    
    # W&B arguments
    parser.add_argument('--wandb_mode', type=str, default='online', 
                        choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_project', type=str, default='thesis-uad')
    parser.add_argument('--wandb_entity', type=str, 
                        default=None)
    
    args = parser.parse_args()

    # Setup logging
    log_dir = os.path.join(args.output_dir, args.experiment_name, 'logs')
    logger = get_logger(args.experiment_name, log_dir)
    print_fn = logger.info

    # Setup device
    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    print_fn(f'Device: {device}')
    print_fn(f'Experiment: {args.experiment_name}')
    print_fn(f'NIfTI directories: {args.nifti_dirs}')
    print_fn(f'Slice range: {args.min_slice_pct}% - {args.max_slice_pct}%')

    # Initialize W&B
    if WANDB_AVAILABLE and args.wandb_mode != 'disabled':
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.experiment_name,
            mode=args.wandb_mode,
            config={
                'experiment_name': args.experiment_name,
                'nifti_dirs': args.nifti_dirs,
                'total_iters': args.total_iters,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'noise_lambda': args.noise_lambda,
                'min_slice_pct': args.min_slice_pct,
                'max_slice_pct': args.max_slice_pct,
                'cached': args.cached,
                'seed': 111,
            },
            tags=['SCRD4AD', 'training', 'nifti', args.experiment_name],
        )

    train(args, print_fn, device)
