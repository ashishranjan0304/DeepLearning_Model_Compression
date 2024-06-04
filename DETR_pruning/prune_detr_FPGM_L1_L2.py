# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from thop import profile


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import datasets               
import util.misc as utils
from util.plot_utils import plot_logs
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model



import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
#from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, timing
import models
import numpy as np
import pickle
from scipy.spatial import distance
import pdb

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
    # Pruning Argument
    parser.add_argument("--rate_norm", type=float, default=1, help="Normalization rate")
    parser.add_argument("--rate_dist", type=float, default=0, help="Distance rate")
    parser.add_argument("--dist_type", type=str, default="l2", help="Type of distance")
    parser.add_argument("--layer_begin", type=int, default=10, help="Beginning layer index")
    parser.add_argument("--layer_end", type=int, default=120, help="Ending layer index")
    parser.add_argument("--layer_inter", type=int, default=1, help="Layer interval")
    parser.add_argument("--epoch_prune", type=int, default=1, help="Epoch for pruning")
    parser.add_argument("--use_cuda", default=torch.cuda.is_available(), help="Use CUDA if available")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--print_freq", type=int, default=1, help="Print frequency")
    return parser

def calculate_sparsity(module):
    """Calculate the sparsity of a module."""
    if hasattr(module, "weight"):
        total_weights = module.weight.nelement()
        zero_weights = torch.sum(module.weight == 0).item()
        sparsity = 100.0 * zero_weights / total_weights
        return sparsity
    return None


def show_sparsity(model):
    """Display the sparsity for the model."""
    print("Sparsity in the model:")
    for name, module in model.named_modules():
        sparsity = calculate_sparsity(module)
        if sparsity is not None:
            print(f"{name}: {sparsity:.2f}%")

def calculate_parameters_and_size(net):
    total_trainable = 0
    total_params = 0
    pruned_params = 0
    model_size = 0
    pruned_model_size = 0
    
    print('Trainable parameters:')
    for name, param in net.named_parameters():
        param_size = param.numel() * param.element_size()
        model_size += param_size  # Calculate the size of each parameter
        total_params += param.numel()  # Count all parameters
        if param.requires_grad:
            print(name, '\t', param.numel())
            total_trainable += param.numel()
    
    print()
    print('Total trainable parameters:', total_trainable)
    
    # Check pruned parameters and calculate pruned model size
    for name, module in net.named_modules():
        # Skip the root module itself
        if name == '':
            continue
        
        if hasattr(module, 'weight') and module.weight is not None:
            total_elements = module.weight.nelement()
            pruned_elements = torch.sum(module.weight == 0).item()
            pruned_params += pruned_elements
            pruned_model_size += (total_elements - pruned_elements) * module.weight.element_size()
        
        if hasattr(module, 'bias') and module.bias is not None:
            total_elements = module.bias.nelement()
            pruned_elements = torch.sum(module.bias == 0).item()
            pruned_params += pruned_elements
            pruned_model_size += (total_elements - pruned_elements) * module.bias.element_size()
    
    remaining_params = total_params - pruned_params
    
    print()
    print("Parameters pruned:", pruned_params)
    print("Remaining parameters:", remaining_params)
    print("Total parameters:", total_params)
    
    # Convert model size to MB
    model_size_MB = model_size / (1024 ** 2)
    pruned_model_size_MB = pruned_model_size / (1024 ** 2)
    print("Model size (MB):", model_size_MB)
    print("Model size after pruning (MB):", pruned_model_size_MB)

def calculate_fps(net, input_shape, device='cpu', num_iterations=100):
    net.to(device)
    net.eval()
    total_time = 0.0

    for _ in range(num_iterations):
        input_data = torch.randn(*input_shape).to(device)
        start_time = time.time()
        with torch.no_grad():
            net(input_data)
        end_time = time.time()
        total_time += end_time - start_time

    avg_fps = num_iterations / total_time
    return avg_fps

class Mask:
    def __init__(self, model, args):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.distance_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.filter_small_index = {}
        self.filter_large_index = {}
        self.similar_matrix = {}
        self.norm_matrix = {}
        self.args = args

    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        #print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

            #print("filter codebook done")
        else:
            pass
        return codebook

    def get_filter_index(self, weight_torch, compress_rate, length):
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            # print("filter index done")
        else:
            pass
        return filter_small_index, filter_large_index

    # optimize for fast ccalculation
    def get_filter_similar(self, weight_torch, compress_rate, distance_rate, length, dist_type="l2"):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)

            if dist_type == "l2" or "cos":
                norm = torch.norm(weight_vec, 2, 1)
                norm_np = norm.cpu().numpy()
            elif dist_type == "l1":
                norm = torch.norm(weight_vec, 1, 1)
                norm_np = norm.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm_np.argsort()[filter_pruned_num:]
            filter_small_index = norm_np.argsort()[:filter_pruned_num]
            indices = torch.LongTensor(filter_large_index).cuda()
            weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()
            # for euclidean distance
            if dist_type == "l2" or "l1":
                similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
            elif dist_type == "cos":  # for cos similarity
                similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
            #print("similar index done")
        else:
            pass
        return codebook

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, rate_norm_per_layer, rate_dist_per_layer):
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate[index] = 1
            self.distance_rate[index] = 1
        for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
            self.compress_rate[key] = rate_norm_per_layer
            self.distance_rate[key] = rate_dist_per_layer

        # to jump the last fc layer
        self.mask_index = [x for x in range(self.args.layer_begin, self.args.layer_end, self.args.layer_inter)]
    #        self.mask_index =  [x for x in range (0,330,3)]

    def init_mask(self, rate_norm_per_layer, rate_dist_per_layer, dist_type):
        self.init_rate(rate_norm_per_layer, rate_dist_per_layer)
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                # mask for norm criterion
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                if args.use_cuda:
                    self.mat[index] = self.mat[index].cuda()

                # # get result about filter index
                # self.filter_small_index[index], self.filter_large_index[index] = \
                #     self.get_filter_index(item.data, self.compress_rate[index], self.model_length[index])

                # mask for distance criterion
                self.similar_matrix[index] = self.get_filter_similar(item.data, self.compress_rate[index],
                                                                     self.distance_rate[index],
                                                                     self.model_length[index], dist_type=dist_type)
                self.similar_matrix[index] = self.convert2tensor(self.similar_matrix[index])
                if args.use_cuda:
                    self.similar_matrix[index] = self.similar_matrix[index].cuda()
        print("mask Ready")

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                #print(f"Parameter index: {index}, item.data shape: {item.data.shape}, expected shape: {self.model_length[index]}")
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        print("mask Done")


    def do_similar_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.similar_matrix[index]
                item.data = b.view(self.model_size[index])
        print("mask similar Done")

    def do_grad_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.grad.data.view(self.model_length[index])
                # reverse the mask of model
                # b = a * (1 - self.mat[index])
                b = a * self.mat[index]
                b = b * self.similar_matrix[index]
                item.grad.data = b.view(self.model_size[index])
        # print("grad zero Done")
    def if_zero(self):
        for index, (name, item) in enumerate(self.model.named_parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()
                nonzero_count = np.count_nonzero(b)
                zero_count = len(b) - nonzero_count
                print(f"Module name: {name}, Nonzero weights: {nonzero_count}, Zero weights: {zero_count}")


def calculate_flops(model, input_shape):
    input_data = torch.randn(*input_shape)
    flops, params = profile(model, inputs=(input_data,))
    gflops=flops_g = flops / 10**9
    return gflops,params

# Main function for DETR training and evaluation with pruning
def main(args):
    device = torch.device(args.device)

    utils.init_distributed_mode(args)  # Ensure distributed training setup

    # Set seeds for reproducibility
    seed = args.seed + utils.get_rank()  # Ensure different seeds for each rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #device = torch.device(args.device)

    # Build the model and criterion
    model, criterion, postprocessors = build_model(args)
    model.to(device)



    # Check if distributed mode is enabled and set up DDP accordingly
    if args.distributed:
        # Ensure correct device assignment
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
        model_without_ddp = model.module  # Access the underlying model
    else:
        model_without_ddp = model

    mask = Mask(model_without_ddp.backbone[0].body, args)
    mask.init_length()
    mask.model = model_without_ddp.backbone[0].body
    mask.init_mask(args.rate_norm, args.rate_dist, args.dist_type)
    mask.do_mask()
    mask.do_similar_mask()
    #net = mask.model


    # Distributed data parallel setup
    if args.distributed:
        # Correct device assignment for distributed setup
        model_without_ddp = model.module  # Ensure proper reference

    # Set up optimizer and learning rate scheduler
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr_backbone},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)

    # Build datasets and data loaders
    dataset_train = build_dataset("train", args)
    dataset_val = build_dataset("val", args)

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
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)
        
    # Resume from checkpoint if provided
    if args.resume:
        checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True) if args.resume.startswith("https") else torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)  # 'strict=False' to avoid shape mismatches
        if not args.eval and "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        # if args.output_dir:
        #     utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        input_shape = (2, 3, 800, 800)  # Example input shape, adjust according to your model
        # Calculate FPS
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Calculating FPS...")
        FPS = calculate_fps(model_without_ddp, input_shape, device=device)
        print("Average FPS:", FPS)
        # Calculate and show the final sparsity after training
        show_sparsity(model_without_ddp)
        #Total number of parameters after pruning
        calculate_parameters_and_size(model_without_ddp)
        return

    # Start training
    best_ap = 0.0  # Initialize the variable to keep track of the highest AP
    print("Start training")
    start_time = time.time()
    prune_amount = 0.05  # Example prune rate
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)


#THIS FORM FPGM
        if epoch % args.epoch_prune == 0 or epoch == args.epochs - 1:
            mask.model = model_without_ddp.backbone[0].body
            mask.if_zero()
            mask.init_mask(args.rate_norm, args.rate_dist, args.dist_type)
            mask.do_mask()
            mask.do_similar_mask()
            mask.if_zero()

        # Training step
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()  # Step the learning rate scheduler

        # Save checkpoint and evaluate
        if args.output_dir:
            checkpoint_paths = [Path(args.output_dir) / "checkpoint.pth"]
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(Path(args.output_dir) / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        # Evaluation step
        base_ds = get_coco_api_from_dataset(dataset_val)  # Ensure base dataset for evaluation
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

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

                
        # Logging
        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, **{f"test_{k}": v for k, v in test_stats.items()}, "epoch": epoch}
        if args.output_dir and utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            plot_logs([Path(args.output_dir)], fields=['class_error', 'loss', 'mAP'], output_dir=args.output_dir)
   
    

    # Final training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time: {total_time_str}")

    # Calculate and show the final sparsity after training
    show_sparsity(model_without_ddp)
    #Total number of parameters after pruning



# Script entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser("DETR training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)



