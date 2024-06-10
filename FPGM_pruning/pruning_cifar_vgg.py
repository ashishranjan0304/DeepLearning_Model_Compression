from __future__ import print_function
import argparse
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os, sys, shutil, time, random
from scipy.spatial import distance
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#import models
from models.vgg_cifar10 import vgg

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('data_path', type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, default='cifar100', help='training dataset (default: cifar100)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N', help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--output_dir', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str, help='architecture to use')
parser.add_argument('--depth', default=16, type=int, help='depth of the neural network')
# compress rate
parser.add_argument('--rate_norm', type=float, default=1, help='the remaining ratio of pruning based on Norm')
parser.add_argument('--rate_dist', type=float, default=0, help='the reducing ratio of pruning based on Distance')

# compress parameter
parser.add_argument('--layer_begin', type=int, default=1, help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=1, help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=1, help='compress layer of model')
parser.add_argument('--epoch_prune', type=int, default=1, help='compress layer of model')
parser.add_argument('--dist_type', default='l2', type=str, choices=['l2', 'l1', 'cos'], help='distance type of GM')

# pretrain model
parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', help='use state dcit or not')
parser.add_argument('--use_pretrain', dest='use_pretrain', action='store_true', help='use pre-trained model or not')
parser.add_argument('--pretrain_path', default='', type=str, help='..path of pre-trained model')
parser.add_argument('--use_precfg', dest='use_precfg', action='store_true', help='use precfg or not')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--arch_change', action='store_true', help='prune, save and evaluate model on validation set after architecture change')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

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

def calculate_fps(net, input_shape, device='cpu', num_iterations=10000):
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

import torch
#import torch_pruning as tp

def prune_and_save_model(pruned_model, save_path, device='cpu'):
    """
    Prunes channels with all-zero weights from Conv2d layers of the given model and saves the pruned model.

    Parameters:
    pruned_model (torch.nn.Module): The model to be pruned.
    save_path (str): The path to save the pruned model.
    device (str): The device to use for model pruning (default is 'cpu').

    Returns:
    None
    """
    # Set model to evaluation mode
    pruned_model.eval()
    pruned_model.to(device)

    for name, module in pruned_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            channel_indices = []  # Stores indices of the channels to prune within this conv layer
            t = module.weight.clone().detach()
            t = t.reshape(t.shape[0], -1)
            z = torch.all(t == 0, dim=1)
            z = z.tolist()
            
            for i, flag in enumerate(z):
                if flag:
                    channel_indices.append(i)

            if not channel_indices:
                continue

            # 1. Build dependency graph
            DG = tp.DependencyGraph().build_dependency(pruned_model, example_inputs=torch.randn(1, 3, 32, 32).to(device))

            # 2. Specify the to-be-pruned channels
            group = DG.get_pruning_group(module, tp.prune_conv_out_channels, idxs=channel_indices)

            # 3. Prune all grouped layers that are coupled with the conv layer (included)
            if DG.check_pruning_group(group):  # Avoid full pruning, i.e., channels=0
                group.prune()

    # 4. Save the pruned model
   # pruned_model.zero_grad()  # We don't want to store gradient information
    torch.save(pruned_model, save_path)
    return pruned_model

# Example usage
# pruned_model = ...  # Your model here
# prune_and_save_model(pruned_model, './vgg_cifar10_arch_pruned_net.pth', device='cuda' if torch.cuda.is_available() else 'cpu')



def main():
    # Init logger
    args.save_path=args.output_dir
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

    # Initialize dictionaries to store losses and accuracies
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    if args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_path, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Pad(4),
                                 transforms.RandomCrop(32),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                             ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.data_path, train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.Pad(4),
                                  transforms.RandomCrop(32),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                              ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.data_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    print_log("=> creating model '{}'".format(args.arch), log)
    model = vgg(args.dataset, depth=args.depth)
    print_log("=> network :\n {}".format(model), log)

    if args.cuda:
        model.cuda()

    if args.use_pretrain:
        if os.path.isfile(args.pretrain_path):
            print_log("=> loading pretrain model '{}'".format(args.pretrain_path), log)
        else:
            dir = '/home/yahe/compress/filter_similarity/logs/main_2'
            args.pretrain_path = dir + '/checkpoint.pth.tar'
            print_log("Pretrain path: {}".format(args.pretrain_path), log)
        pretrain = torch.load(args.pretrain_path)
        if args.use_state_dict:
            model.load_state_dict(pretrain['state_dict'])
        else:
            model = pretrain['state_dict']
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model = vgg(dataset='cifar10', depth=16, cfg=checkpoint['cfg'])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(args.resume, checkpoint['epoch'], best_prec1))
            if args.cuda:
                model = model.cuda()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        time1 = time.time()
        test_acc, test_loss = test(test_loader, model, log)
        time2 = time.time()
        print('function took %0.3f ms' % ((time2 - time1) * 1000.0))
        calculate_parameters_and_size(model)
        input_shape = (2, 3, 32, 32)  # Example input shape, adjust according to your model
        # Calculate FPS
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Calculating FPS after pruning...")
        fps = calculate_fps(model, input_shape, device=device)
        print("Average FPS :", fps)
        if args.arch_change:
            print("................. After architecture change ...............")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            pruned_model = prune_and_save_model(model, os.path.join(args.save_path, 'pruned_arch.pth'), device=device)
            
            time1 = time.time()
            test_acc, test_loss = test(test_loader, pruned_model, log)
            time2 = time.time()
            print('Function took %0.3f ms' % ((time2 - time1) * 1000.0))
            
            calculate_parameters_and_size(pruned_model)
            
            input_shape = (2, 3, 32, 32)  # Example input shape, adjust according to your model
            print("Calculating FPS after pruning and architecture change...")
            fps = calculate_fps(pruned_model, input_shape, device=device)
            print("Average FPS after architecture change:", fps)
        return

    m = Mask(model)
    m.init_length()
    print("-" * 10 + "one epoch begin" + "-" * 10)
    print("remaining ratio of pruning : Norm is %f" % args.rate_norm)
    print("reducing ratio of pruning : Distance is %f" % args.rate_dist)
    print("total remaining ratio is %f" % (args.rate_norm - args.rate_dist))

    val_acc_1, val_loss_1 = test(test_loader, model, log)

    print(" accu before is: %.3f %%" % val_acc_1)

    m.model = model

    m.init_mask(args.rate_norm, args.rate_dist, args.dist_type)
    m.do_mask()
    m.do_similar_mask()
    model = m.model
    if args.cuda:
        model = model.cuda()
    val_acc_2, val_loss_2 = test(test_loader, model, log)
    print(" accu after is: %s %%" % val_acc_2)

    best_prec1 = 0.
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in [args.epochs * 0.5, args.epochs * 0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        train_loss, train_acc = train(train_loader, model, optimizer, epoch, log)
        print(train_loss)
        print("train loss")
        test_acc, test_loss = test(test_loader, model, log)

        # Save losses and accuracies
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)


        if epoch % args.epoch_prune == 0 or epoch == args.epochs - 1:
            m.model = model
            m.if_zero()
            m.init_mask(args.rate_norm, args.rate_dist, args.dist_type)
            m.do_mask()
            m.do_similar_mask()
            m.if_zero()
            model = m.model
            if args.cuda:
                model = model.cuda()
            test_acc, test_loss = test(test_loader, model, log)
        is_best = test_acc > best_prec1
        best_prec1 = max(test_acc, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'cfg': model.cfg
        }, is_best, filepath=args.save_path)

        # Plot and save the curves
        plot_and_save_curves(history, args.save_path, epoch)

def train(train_loader, model, optimizer, epoch, log, m=0):
    model.train()
    total_loss = 0.  # For accumulating loss over all batches
    correct = 0  # For accumulating number of correct predictions
    total_samples = 0  # For accumulating total number of samples

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # Accumulate loss and correct predictions
        total_loss += loss.item() * data.size(0)  # Multiply by batch size to get total loss for the batch
        pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)  # Accumulate the number of samples
        
        if batch_idx % args.log_interval == 0:
            print_log('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), log)

    avg_loss = total_loss / total_samples
    train_acc = correct / total_samples

    print_log('Train Epoch: {} \tAverage Loss: {:.6f}, Accuracy: {}/{} ({:.1f}%)'.format(
        epoch, avg_loss, correct, total_samples,
        100. * train_acc), log)
    
    return avg_loss, train_acc

def test(test_loader, model, log):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    correct = correct.item()
    accuracy = correct / len(test_loader.dataset)
    print_log('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * accuracy), log)
    return accuracy, test_loss

def plot_and_save_curves(history, save_path, epoch):
    epochs = range(1, epoch + 2)
    plt.figure()

    # Plot training and testing loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in history['train_loss']], 'b-', label='Train Loss')
    plt.plot(epochs, [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in history['test_loss']], 'r-', label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and testing accuracy
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    plt.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'training_plots.png'))
    plt.close()

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
        self.cfg = [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256]

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
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

            # print("filter codebook done")
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

    def get_filter_similar_old(self, weight_torch, compress_rate, distance_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]
            print('weight_vec.size', weight_vec.size())
            # distance using pytorch function
            similar_matrix = torch.zeros((len(filter_large_index), len(filter_large_index)))
            for x1, x2 in enumerate(filter_large_index):
                for y1, y2 in enumerate(filter_large_index):
                    # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                    # similar_matrix[x1, y1] = cos(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0]
                    pdist = torch.nn.PairwiseDistance(p=2)
                    # print('weight_vec[x2].size', weight_vec[x2].size())
                    similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]
                    # print('weight_vec[x2].size after', weight_vec[x2].size())
            # more similar with other filter indicates large in the sum of row
            similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

            # for cos similar: get the filter index with largest similarity
            # similar_pruned_num = len(similar_sum) - similar_pruned_num
            # similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            # similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            # similar_index_for_filter = [filter_large_index[i] for i in similar_large_index]

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            print('filter_large_index', filter_large_index)
            print('filter_small_index', filter_small_index)
            print('similar_sum', similar_sum)
            print('similar_large_index', similar_large_index)
            print('similar_small_index', similar_small_index)
            print('similar_index_for_filter', similar_index_for_filter)
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
            print("similar index done")
        else:
            pass
        return codebook

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

            # # distance using pytorch function
            # similar_matrix = torch.zeros((len(filter_large_index), len(filter_large_index)))
            # for x1, x2 in enumerate(filter_large_index):
            #     for y1, y2 in enumerate(filter_large_index):
            #         # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            #         # similar_matrix[x1, y1] = cos(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0]
            #         pdist = torch.nn.PairwiseDistance(p=2)
            #         similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]
            # # more similar with other filter indicates large in the sum of row
            # similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

            # distance using numpy function
            indices = torch.LongTensor(filter_large_index).cuda()
            weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()
            # for euclidean distance
            if dist_type == "l2" or "l1":
                similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
            elif dist_type == "cos":  # for cos similarity
                similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # print('similar_matrix 1',similar_matrix.cpu().numpy())
            # print('similar_matrix 2', similar_matrix_2)
            # # print('similar_matrix 3', similar_matrix_3)
            # result = np.absolute(similar_matrix.cpu().numpy() - similar_matrix_2)
            # print('result',result)
            # print('similar_matrix',similar_matrix.cpu().numpy())
            # print('similar_matrix_2', similar_matrix_2)
            # print('result', similar_matrix.cpu().numpy()-similar_matrix_2)
            # print('similar_sum',similar_sum)
            # print('similar_sum_2', similar_sum_2)
            # print('result sum', similar_sum-similar_sum_2)

            # for cos similar: get the filter index with largest similarity
            # similar_pruned_num = len(similar_sum) - similar_pruned_num
            # similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            # similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            # similar_index_for_filter = [filter_large_index[i] for i in similar_large_index]

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            # print('filter_large_index', filter_large_index)
            # print('filter_small_index', filter_small_index)
            # print('similar_sum', similar_sum)
            # print('similar_large_index', similar_large_index)
            # print('similar_small_index', similar_small_index)
            # print('similar_index_for_filter', similar_index_for_filter)
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

    def init_rate(self, rate_norm_per_layer, rate_dist_per_layer, pre_cfg=True):
        if args.arch == 'vgg':
            cfg = [32, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256]
            cfg_index = 0
            for index, item in enumerate(self.model.named_parameters()):
                self.compress_rate[index] = 1
                self.distance_rate[index] = 1
                if len(item[1].size()) == 4:
                   # print(item[1].size())
                    if not pre_cfg:
                        self.compress_rate[index] = rate_norm_per_layer
                        self.distance_rate[index] = rate_dist_per_layer
                        self.mask_index.append(index)
                       # print(item[0], "self.mask_index", self.mask_index)
                    else:
                        self.compress_rate[index] = rate_norm_per_layer
                        self.distance_rate[index] = 1 - cfg[cfg_index] / item[1].size()[0]
                        self.mask_index.append(index)
                       # print(item[0], "self.mask_index", self.mask_index, cfg_index, cfg[cfg_index], item[1].size()[0],
                        self.distance_rate[index],
                        #print("self.distance_rate", self.distance_rate)
                        cfg_index += 1
        # for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
        #     self.compress_rate[key] = rate_norm_per_layer
        #     self.distance_rate[key] = rate_dist_per_layer
        # different setting for  different architecture
        # if args.arch == 'resnet20':
        #     last_index = 57
        # elif args.arch == 'resnet32':
        #     last_index = 93
        # elif args.arch == 'resnet56':
        #     last_index = 165
        # elif args.arch == 'resnet110':
        #     last_index = 327
        # # to jump the last fc layer
        # self.mask_index = [x for x in range(0, last_index, 3)]

    def init_mask(self, rate_norm_per_layer, rate_dist_per_layer, dist_type):
        self.init_rate(rate_norm_per_layer, rate_dist_per_layer, pre_cfg=args.use_precfg)
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                # mask for norm criterion
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                if args.cuda:
                    self.mat[index] = self.mat[index].cuda()

                # # get result about filter index
                # self.filter_small_index[index], self.filter_large_index[index] = \
                #     self.get_filter_index(item.data, self.compress_rate[index], self.model_length[index])

                # mask for distance criterion
                self.similar_matrix[index] = self.get_filter_similar(item.data, self.compress_rate[index],
                                                                     self.distance_rate[index],
                                                                     self.model_length[index], dist_type=dist_type)
                self.similar_matrix[index] = self.convert2tensor(self.similar_matrix[index])
                if args.cuda:
                    self.similar_matrix[index] = self.similar_matrix[index].cuda()
        #print("mask Ready")

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        #print("mask Done")

    def do_similar_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.similar_matrix[index]
                item.data = b.view(self.model_size[index])
        #print("mask similar Done")

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
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                # if index == 0:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()
                print(
                    "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))


if __name__ == '__main__':
    main()
