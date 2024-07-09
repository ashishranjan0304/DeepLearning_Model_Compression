import argparse
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.autograd import Variable
import os, sys, shutil, time, random
from scipy.spatial import distance
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models
from data_loading import CarvanaDataset, BasicDataset
from tqdm import tqdm
from dice_score import multiclass_dice_coeff, dice_coeff, dice_loss
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CARVANA Images training')
parser.add_argument('data_path', type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, default='cifar100', help='training dataset (default: cifar100)')
parser.add_argument('--batch_size', type=int, default=3, metavar='N', help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N', help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=160, metavar='N', help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--save_path', default='./logs', type=str, metavar='PATH', help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='UNet', type=str, help='architecture to use')
parser.add_argument('--depth', default=16, type=int, help='depth of the neural network')
parser.add_argument('--rate_norm', type=float, default=0.9, help='the remaining ratio of pruning based on Norm')
parser.add_argument('--rate_dist', type=float, default=0.1, help='the reducing ratio of pruning based on Distance')
parser.add_argument('--layer_begin', type=int, default=1, help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=1, help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=1, help='compress layer of model')
parser.add_argument('--epoch_prune', type=int, default=1, help='compress layer of model')
parser.add_argument('--dist_type', default='l2', type=str, choices=['l2', 'l1', 'cos'], help='distance type of GM')
parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', help='use state dict or not')
parser.add_argument('--use_pretrain', dest='use_pretrain', action='store_true', help='use pre-trained model or not')
parser.add_argument('--pretrain_path', default='', type=str, help='path of pre-trained model')
parser.add_argument('--use_precfg', dest='use_precfg', action='store_true', help='use precfg or not')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs to use for training (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}


def plot_metrics(metrics, fps, best_ap, output_dir, dice_scores, times, times_per_image):
    groups = {
        'Parameters': [
            ('Total Trainable Parameters', metrics['total_trainable']),
            ('Pruned Parameters', metrics['pruned_params']),
            ('Remaining Parameters', metrics['remaining_params'])
        ],
        'Model Size (MB)': [
            ('Model Size (MB)', metrics['model_size_MB']),
            ('Pruned Model Size (MB)', metrics['pruned_model_size_MB'])
        ],
        'Performance': [
            ('Average FPS', fps)
        ],
        'Validation Metrics': [
            ('Best AP (%)', best_ap)
        ]
    }

    colors = [
        'skyblue', 'lightgreen', 'salmon', 'lightcoral',
        'mediumpurple', 'orange', 'gold', 'lightpink',
        'turquoise', 'yellowgreen', 'cyan', 'magenta'
    ]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 18))
    fig.suptitle('Model Metrics After Pruning', fontsize=20)

    color_idx = 0

    for ax, (group_name, group_metrics) in zip(axes.flatten()[:4], groups.items()):
        labels, values = zip(*group_metrics)
        values = [v.cpu().item() if torch.is_tensor(v) else v for v in values]  # Ensure values are on CPU and converted to numbers

        bar_colors = colors[color_idx:color_idx+len(labels)]
        bars = ax.barh(range(len(labels)), values, color=bar_colors)
        ax.set_title(group_name, fontsize=16)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, rotation=0, ha='right')

        for bar, label in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2.0, f'{label:.2f}', va='center', ha='left', fontsize=12, color='black')

        color_idx += len(labels)

    # Convert dice scores and times to CPU numpy arrays before plotting
    dice_scores = [score.cpu().numpy() if torch.is_tensor(score) else score for score in dice_scores]
    times = [time.cpu().numpy() if torch.is_tensor(time) else time for time in times]
    times_per_image = [time.cpu().numpy() if torch.is_tensor(time) else time for time in times_per_image]

    # Plot dice scores
    axes[2, 0].plot(dice_scores, label='Dice Score')
    axes[2, 0].set_title('Dice Score over Epochs')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Dice Score')
    axes[2, 0].legend()

    # Plot time per image
    axes[2, 1].plot(times, label='Total Time')
    axes[2, 1].plot(times_per_image, label='Time per Image')
    axes[2, 1].set_title('Time Metrics')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Time (s)')
    axes[2, 1].legend()

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'model_metrics.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved at {plot_path}")



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
    
    for name, param in net.named_parameters():
        param_size = param.numel() * param.element_size()
        model_size += param_size  # Calculate the size of each parameter
        total_params += param.numel()  # Count all parameters
        if param.requires_grad:
            total_trainable += param.numel()
    
    for name, module in net.named_modules():
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
    
    model_size_MB = model_size / (1024 ** 2)
    pruned_model_size_MB = pruned_model_size / (1024 ** 2)
    
    return {
        'total_trainable': total_trainable,
        'pruned_params': pruned_params,
        'remaining_params': remaining_params,
        'model_size_MB': model_size_MB,
        'pruned_model_size_MB': pruned_model_size_MB
    }


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


def plot_and_save_predictions(image, mask_true, mask_pred, save_path, epoch, batch_idx):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Original Image')
    if mask_true.ndim == 2 or mask_true.shape[0] == 1:
        axes[1].imshow(mask_true.cpu().numpy().squeeze(), cmap='gray')
    else:
        axes[1].imshow(mask_true.argmax(0).cpu().numpy(), cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    if mask_pred.ndim == 2 or mask_pred.shape[0] == 1:
        axes[2].imshow(mask_pred.cpu().numpy().squeeze(), cmap='gray')
    else:
        axes[2].imshow(mask_pred.argmax(0).cpu().numpy(), cmap='gray')
    axes[2].set_title('Predicted Mask')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'prediction_epoch{epoch}_batch{batch_idx}.png'))
    plt.close()

@torch.no_grad()
def evaluate(net, dataloader, device, amp, save_path=None, epoch=0):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    total_time = 0

    model = net.module  # Access the underlying model

    if hasattr(torch, 'autocast'):
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            for batch_idx, batch in enumerate(tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False)):
                image, mask_true = batch['image'], batch['mask']

                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                mask_true = mask_true.to(device=device, dtype=torch.long)

                start_time = time.time()
                mask_pred = net(image)
                end_time = time.time()
                total_time += end_time - start_time

                if model.n_classes == 1:
                    assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                    mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                    dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                else:
                    assert mask_true.min() >= 0 and mask_true.max() < model.n_classes, 'True mask indices should be in [0, n_classes['
                    mask_true = F.one_hot(mask_true, model.n_classes).permute(0, 3, 1, 2).float()
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
                    dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

                if save_path is not None:
                    plot_and_save_predictions(image[0], mask_true[0], mask_pred[0], save_path, epoch, batch_idx)
    else:
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False)):
            image, mask_true = batch['image'], batch['mask']

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            start_time = time.time()
            mask_pred = net(image)
            end_time = time.time()
            total_time += end_time - start_time

            if model.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < model.n_classes, 'True mask indices should be in [0, n_classes['
                mask_true = F.one_hot(mask_true, model.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

            if save_path is not None:
                plot_and_save_predictions(image[0], mask_true[0], mask_pred[0], save_path, epoch, batch_idx)

    net.train()
    avg_time_per_image = total_time / num_val_batches
    return dice_score / max(num_val_batches, 1), total_time, avg_time_per_image


def main():
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.seed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.seed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Norm Pruning Rate: {}".format(args.rate_norm), log)
    print_log("Distance Pruning Rate: {}".format(args.rate_dist), log)
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    print_log("use pretrain: {}".format(args.use_pretrain), log)
    print_log("Pretrain path: {}".format(args.pretrain_path), log)
    print_log("Dist type: {}".format(args.dist_type), log)
    print_log("Pre cfg: {}".format(args.use_precfg), log)

    if args.dataset == 'CARVANA':
        try:
            dataset = CarvanaDataset(args.data_path + '/imgs', args.data_path + '/masks', scale=1)
        except (AssertionError, RuntimeError, IndexError):
            dataset = BasicDataset(args.data_path + '/imgs', args.data_path + '/masks', scale=1)
        dataset = torch.utils.data.Subset(dataset, range(0, 100))

        val_percent = 0.2
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        
        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set)
        
        loader_args = dict(batch_size=args.batch_size // torch.distributed.get_world_size(), num_workers=os.cpu_count(), pin_memory=True)
        train_loader = torch.utils.data.DataLoader(train_set, sampler=train_sampler, **loader_args)
        test_loader = torch.utils.data.DataLoader(val_set, sampler=val_sampler, **loader_args)

    print_log("=> creating model '{}'".format(args.arch), log)
    model = models.__dict__[args.arch](n_channels=3, n_classes=2)

    model = model.to(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    print_log("=> network :\n {}".format(model), log)

    if args.use_pretrain:
        if os.path.isfile(args.pretrain_path):
            print_log("=> loading pretrain model '{}'".format(args.pretrain_path), log)
        else:
            dir = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/anisha-gpu/code/Users/Anisha.Gupta/model-pruning-fpgm-unet-copy/logs/unet_pretrain/prune_precfg_epoch40_varience1'
            args.pretrain_path = dir + '/MODEL.pth'
            print_log("Pretrain path: {}".format(args.pretrain_path), log)

        pretrain = torch.load(args.pretrain_path, map_location='cpu')
        if args.use_state_dict:
            model.module.load_state_dict(pretrain)  # Access the underlying model
        else:
            model = pretrain

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            # args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.module.load_state_dict(checkpoint['state_dict'])  # Access     the underlying model
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}".format(args.resume, checkpoint['epoch'], best_prec1))

    if args.evaluate:
        time1 = time.time()
        evaluate(model, test_loader, device, amp=False)
        time2 = time.time()
        print('function took %0.3f ms' % ((time2 - time1) * 1000.0))
        return

    m = Mask(model.module)  # Access the underlying model
    m.init_length()
    print("-" * 10 + "one epoch begin" + "-" * 10)
    print("remaining ratio of pruning : Norm is %f" % args.rate_norm)
    print("reducing ratio of pruning : Distance is %f" % args.rate_dist)
    print("total remaining ratio is %f" % (args.rate_norm - args.rate_dist))

    device = torch.device(f'cuda:{local_rank}')
    val_acc_1, _, _ = evaluate(model, test_loader, device, amp=False)
    print(" accu before is: %.3f %%" % val_acc_1.cpu().float())

    m.model = model.module  # Access the underlying model

    m.init_mask(args.rate_norm, args.rate_dist, args.dist_type)
    m.do_mask()
    m.do_similar_mask()
    model.module = m.model  # Access the underlying model

    val_acc_2, _, _ = evaluate(model, test_loader, device, amp=False)
    print(" accu after is: %.3f %%" % val_acc_2.cpu().float())

    best_prec1 = 0.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    amp: bool = False
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.module.n_classes > 1 else nn.BCEWithLogitsLoss()  # Access the underlying model
    torch.cuda.empty_cache()

    dice_scores = []
    times = []
    times_per_image = []

    for epoch in range(args.start_epoch, args.epochs):
        if epoch in [args.epochs * 0.5, args.epochs * 0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        train_sampler.set_epoch(epoch)  # Ensure randomness in shuffling
        train_model(model, train_loader, n_train, optimizer, epoch, log, grad_scaler, device, criterion, dataset)
        
        prec1, total_time, avg_time_per_image = evaluate(model, test_loader, device, amp=False)
        dice_scores.append(prec1.cpu().float())
        times.append(total_time)
        times_per_image.append(avg_time_per_image)
        
        if epoch % args.epoch_prune == 0 or epoch == args.epochs - 1:
            m.model = model.module  # Access the underlying model
            m.if_zero()
            m.init_mask(args.rate_norm, args.rate_dist, args.dist_type)
            m.do_mask()
            m.do_similar_mask()
            m.if_zero()
            model.module = m.model  # Access the underlying model

            val_acc_2, _, _ = evaluate(model, test_loader, device, amp=False)
            val_acc_2 = val_acc_2.to(device)

        is_best = val_acc_2 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        print(" Accuracy after epoch : %.3f %%" % best_prec1.cpu().float())

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),  # Access the underlying model
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, filepath=args.save_path)
    
    evaluate(model, test_loader, device, save_path=args.save_path, amp=False)

    metrics = calculate_parameters_and_size(model)
    fps = calculate_fps(model, input_shape=(args.batch_size, 3, 256, 256), device=device, num_iterations=100)
    best_ap = best_prec1.cpu().float()
    plot_metrics(metrics, fps, best_ap, args.save_path, dice_scores, times, times_per_image)


def train_model(
        model,
        train_loader, n_train,
        optimizer, epoch, log, grad_scaler,
        device, criterion, dataset,
        amp: bool = False,
        gradient_clipping: float = 1.0,
):
    model.train()
    epoch_loss = 0
    with tqdm(total=n_train, desc=f'Epoch {epoch}', unit='img') as pbar:
        for batch in train_loader:
            images, true_masks = batch['image'], batch['mask']

            assert images.shape[1] == model.module.n_channels, \
                f'Network has been defined with {model.module.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            if hasattr(torch, 'autocast'):
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.module.n_classes == 1:  # Access the underlying model
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.module.n_classes).permute(0, 3, 1, 2).float(),  # Access the underlying model
                            multiclass=True
                        )
            else:
                masks_pred = model(images)
                if model.module.n_classes == 1:  # Access the underlying model
                    loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                else:
                    loss = criterion(masks_pred, true_masks)
                    loss += dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, model.module.n_classes).permute(0, 3, 1, 2).float(),  # Access the underlying model
                        multiclass=True
                    )

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            pbar.update(images.shape[0])
            epoch_loss += loss.item()
            pbar.set_postfix(**{'loss (batch)': loss.item()})


def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


class Mask:
    def __init__(self, model):
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
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 1024, 1024, 512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64, 2]

    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

            print("filter codebook done")
        return codebook

    def get_filter_index(self, weight_torch, compress_rate, length):
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
        return filter_small_index, filter_large_index

    def get_filter_similar_old(self, weight_torch, compress_rate, distance_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)

            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]
            similar_matrix = torch.zeros((len(filter_large_index), len(filter_large_index)))
            for x1, x2 in enumerate(filter_large_index):
                for y1, y2 in enumerate(filter_large_index):
                    pdist = torch.nn.PairwiseDistance(p=2)
                    similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]

            similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
        return codebook

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
            filter_large_index = norm_np.argsort()[filter_pruned_num:]
            filter_small_index = norm_np.argsort()[:filter_pruned_num]

            indices = torch.LongTensor(filter_large_index).cuda() if args.cuda else torch.LongTensor(filter_large_index)
            weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()

            if dist_type == "l2" or "l1":
                similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
            elif dist_type == "cos":
                similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
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

    def init_rate(self, rate_norm_per_layer, rate_dist_per_layer, pre_cfg=True):
        if args.arch == 'UNet':
            cfg = [64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024, 512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64, 2]
            cfg_index = 0
            for index, item in enumerate(self.model.named_parameters()):
                self.compress_rate[index] = 1
                self.distance_rate[index] = 1
                if len(item[1].size()) == 4:
                    if not pre_cfg:
                        self.compress_rate[index] = rate_norm_per_layer
                        self.distance_rate[index] = rate_dist_per_layer
                        self.mask_index.append(index)
                    else:
                        self.compress_rate[index] = rate_norm_per_layer
                        self.distance_rate[index] = 1 - cfg[cfg_index] / item[1].size()[0]
                        self.mask_index.append(index)
                        cfg_index += 1

    def init_mask(self, rate_norm_per_layer, rate_dist_per_layer, dist_type):
        self.init_rate(rate_norm_per_layer, rate_dist_per_layer, pre_cfg=args.use_precfg)
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index], self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                if args.cuda:
                    self.mat[index] = self.mat[index].cuda()

                self.similar_matrix[index] = self.get_filter_similar(item.data, self.compress_rate[index], self.distance_rate[index], self.model_length[index], dist_type=dist_type)
                self.similar_matrix[index] = self.convert2tensor(self.similar_matrix[index])
                if args.cuda:
                    self.similar_matrix[index] = self.similar_matrix[index].cuda()
        print("mask Ready")

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
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
                b = a * self.mat[index]
                b = b * self.similar_matrix[index]
                item.grad.data = b.view(self.model_size[index])

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()
                print("number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))


if __name__ == '__main__':
    main()
