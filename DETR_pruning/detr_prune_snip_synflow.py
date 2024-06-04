import argparse
import datetime
import json
import random
import time
import sys
from pathlib import Path
 
import numpy as np
#import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
 
# Add pruning code directory to system path
sys.path.append('../snip_snyflow_grasp_pruning')
 
import util.misc as utils
from util.plot_utils import plot_logs
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from Utils import load_detr
from Utils import generator ,generator_detr
from Utils import metrics
from train import *
from prune import *
 
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--nclasses', default=2, type=int,
                        help='number of classes to train (excluding background)')
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
 
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
 
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
 
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
 
    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
 
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
 
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
   
    # Pruning parameters
    parser.add_argument('--prune', action='store_true', help='Whether to apply pruning')
    parser.add_argument('--prune_epochs', default=10, type=int, help='Number of epochs to prune')
    parser.add_argument('--compression', default='0.5', type=str, help='Compression rate for pruning')
    parser.add_argument('--pruner', default='mag', type=str, help='Pruner type')
    parser.add_argument('--prune_batch_size', default=64, type=int, help='Batch size for pruning data loader')
    parser.add_argument('--prune_dataset_ratio', default=0.1, type=float, help='Dataset ratio for pruning')
    parser.add_argument('--prune_bias', action='store_true', help='Whether to prune bias terms')
    parser.add_argument('--prune_batchnorm', action='store_true', help='Whether to prune batchnorm layers')
    parser.add_argument('--prune_residual', action='store_true', help='Whether to prune residual connections')
    parser.add_argument('--mask_scope', default='global', type=str, help='Mask scope for pruning')
    parser.add_argument('--compression_schedule', default='exponential', type=str, help='Compression schedule for pruning')
    parser.add_argument('--prune_train_mode', action='store_true', help='Whether to train during pruning')
    parser.add_argument('--reinitialize', action='store_true', help='Whether to reinitialize weights after pruning')
    parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle data during pruning')
    parser.add_argument('--invert', action='store_true', help='Whether to invert masks during pruning')
    return parser
 
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(MaskedLinear, self).__init__(*args, **kwargs)
        self.register_buffer('mask', torch.ones_like(self.weight))

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)

class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', torch.ones_like(self.weight))

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.conv2d(input, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class MaskedBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(MaskedBatchNorm2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', torch.ones_like(self.weight))

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.batch_norm(input, self.running_mean, self.running_var, masked_weight, self.bias, self.training, self.momentum, self.eps)

class MaskedLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(MaskedLayerNorm, self).__init__(*args, **kwargs)
        self.register_buffer('mask', torch.ones_like(self.weight))

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.layer_norm(input, self.normalized_shape, masked_weight, self.bias, self.eps)

def replace_layers_with_masked(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            masked_module = MaskedLinear(module.in_features, module.out_features, module.bias is not None)
            masked_module.load_state_dict(module.state_dict(), strict=False)
            setattr(model, name, masked_module)
        elif isinstance(module, nn.Conv2d):
            masked_module = MaskedConv2d(module.in_channels, module.out_channels, module.kernel_size,
                                         module.stride, module.padding, module.dilation, module.groups, module.bias is not None)
            masked_module.load_state_dict(module.state_dict(), strict=False)
            setattr(model, name, masked_module)
        elif isinstance(module, nn.BatchNorm2d):
            masked_module = MaskedBatchNorm2d(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats)
            masked_module.load_state_dict(module.state_dict(), strict=False)
            setattr(model, name, masked_module)
        elif isinstance(module, nn.LayerNorm):
            masked_module = MaskedLayerNorm(module.normalized_shape, module.eps, module.elementwise_affine)
            masked_module.load_state_dict(module.state_dict(), strict=False)
            setattr(model, name, masked_module)
        elif not isinstance(module, nn.MultiheadAttention):  # Add this condition to skip MultiheadAttention
            replace_layers_with_masked(module)


# Usage example:
# Assuming your DETR model is loaded as `model`
# replace_layers_with_masked(model)
# Define the get_masks function
def get_masks(module):
    """Returns an iterator over module masks, yielding the mask."""
    if hasattr(module, 'mask'):
        yield module.mask

# Define the apply_masks function
def apply_masks(model):
    """Applies the mask to the parameters."""
    for module in model.modules():
        if hasattr(module, 'mask'):
            for mask, param in zip(get_masks(module), module.parameters(recurse=False)):
                param.data.mul_(mask)

def calculate_sparsity(model):
    sparsity_dict = {}
    total_params = 0
    sparse_params = 0
    
    for name, module in model.named_modules():
        module_total_params = 0
        module_sparse_params = 0
        has_param = False
        for pname, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                param_total = param.numel()
                param_sparse = (param == 0).sum().item()
                
                module_total_params += param_total
                module_sparse_params += param_sparse
                total_params += param_total
                sparse_params += param_sparse
                has_param = True
                
        if has_param:  # Only include modules with parameters
            if module_total_params > 0:
                module_sparsity = module_sparse_params / module_total_params
            else:
                module_sparsity = 0
            sparsity_dict[name] = module_sparsity
    
    overall_sparsity = sparse_params / total_params
    return sparsity_dict, total_params, sparse_params, overall_sparsity

def print_sparsity(sparsity_dict, total_params, sparse_params, overall_sparsity):
    for module_name, sparsity in sparsity_dict.items():
        print(f"Module: {module_name}, Sparsity: {sparsity:.4f}")
    print(f"Total Trainable Parameters: {total_params}")
    print(f"Sparse Parameters: {sparse_params}")
    print(f"Overall Sparsity: {overall_sparsity:.4f}")
    print(f"Percentage of Pruned Parameters: {overall_sparsity * 100:.2f}%")

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
 
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)
 
    device = torch.device(args.device)
 
    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
 
    model, criterion, postprocessors = build_model(args)
    replace_layers_with_masked(model)
    model.to(device)
 
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
 
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr_backbone},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
 
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
 
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
 
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
 
    if args.dataset_file == "coco_panoptic":
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)
 
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
 
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
 
    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        
        # Apply masks to parameters
        #apply_masks(model)

        # Calculate and print the sparsity
        sparsity_dict, total_params, sparse_params, overall_sparsity = calculate_sparsity(model)
        print_sparsity(sparsity_dict, total_params, sparse_params, overall_sparsity)

        return
    #print(model)
    for name, param in model.named_parameters():
        if 'attention' in name.lower():
            print(name, param.shape)

    best_ap = 0.0  # Initialize the variable to keep track of the highest AP
    print("Start training")
    start_time = time.time()
 
    if args.prune:
        # Prune the model
        #prune_loader = load.dataloader(args.dataset_file, args.prune_batch_size, True, args.num_workers, args.prune_dataset_ratio * len(dataset_train))
        prune_loader=data_loader_train
        print('Pruning with {} for {} epochs.'.format(args.pruner, args.prune_epochs))
        pruner = load_detr.pruner(args.pruner)(generator_detr.get_masked_parameters(model_without_ddp))
        print(pruner)
        print(type(pruner))
        print("ashish ashish")
        generator_detr.print_masked_parameters_info(model_without_ddp)
        sparsity = 10**(-float(args.compression))
        prune_loop(model_without_ddp, criterion, pruner, prune_loader, device, sparsity, args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize, args.prune_train_mode, args.shuffle, args.invert)
    
    
        # Apply masks to parameters
        apply_masks(model)

        # Calculate and print the sparsity
        sparsity_dict, total_params, sparse_params, overall_sparsity = calculate_sparsity(model)
        print_sparsity(sparsity_dict, total_params, sparse_params, overall_sparsity)

    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch, 'args': args}, checkpoint_path)
 
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir)
 
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch, 'n_parameters': n_parameters}

        # Extract the AP value from coco_eval_bbox
        coco_eval_bbox = test_stats['coco_eval_bbox']
        ap_value = coco_eval_bbox[0]  # Assuming AP at IoU=0.50:0.95 is the first value
        
        # Check if the current AP is better than the best AP seen so far
        if ap_value > best_ap:
            best_ap = ap_value  # Update the best AP
            # Save the model checkpoint
            if args.output_dir:
                best_checkpoint_path = Path(args.output_dir) / "best_checkpoint.pth"
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "test_stats": test_stats
                    },
                    best_checkpoint_path,
                )
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
 
            plot_logs([Path(args.output_dir)], fields=['class_error', 'loss', 'mAP'], output_dir=args.output_dir)

            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval" / name)
 
         # Calculate and print the sparsity
        sparsity_dict, total_params, sparse_params, overall_sparsity = calculate_sparsity(model)
        print_sparsity(sparsity_dict, total_params, sparse_params, overall_sparsity)

    

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
 
 