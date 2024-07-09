import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from Utils import metrics
from train import *
from prune import *
import time


import argparse
import json
import os

import matplotlib.pyplot as plt
import os

def plot_model_metrics(model_metrics, fps, total_time, time_per_image, output_dir):
    groups = {
        'Parameters': [
            ('Total Parameters', model_metrics['total_params']),
            ('Pruned Parameters', model_metrics['pruned_params']),
        ],
        'FLOPs': [
            ('Possible FLOPs', model_metrics['possible_flops']),
            ('Total FLOPs', model_metrics['total_flops'])
        ],
        'Performance': [
            ('Average FPS', fps),
            ('Total Time (s)', total_time),
            ('Time per Image (ms)', time_per_image)
        ],
        'Final Results': [
            ('Train Loss', model_metrics['final_result']['train_loss']),
            ('Test Loss', model_metrics['final_result']['test_loss']),
            ('Train Accuracy', model_metrics['final_result']['train_accuracy1']),
            ('Test Accuracy', model_metrics['final_result']['test_accuracy1']),
        ]
    }

    colors = [
        'skyblue', 'lightgreen', 'salmon', 'lightcoral',
        'mediumpurple', 'orange', 'gold', 'lightpink',
        'turquoise', 'yellowgreen', 'cyan', 'magenta'
    ]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
    fig.suptitle('Model Metrics After Pruning', fontsize=20)

    color_idx = 0

    for ax, (group_name, group_metrics) in zip(axes.flatten(), groups.items()):
        labels, values = zip(*group_metrics)
        values = [v.cpu().item() if torch.is_tensor(v) else v for v in values]  # Ensure values are on CPU and converted to numbers

        bar_colors = colors[color_idx:color_idx+len(labels)]
        bars = ax.barh(range(len(labels)), values, color=bar_colors)
        ax.set_title(group_name, fontsize=16)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, rotation=0, ha='right')

        for i, bar in enumerate(bars):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2.0, f'{values[i]:.2f}', va='center', ha='left', fontsize=12, color='black')

        color_idx += len(labels)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'model_metrics.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved at {plot_path}")


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

def evaluate_model(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    start_time = time.time()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(1, dim=1)
            correct1 += pred.eq(target.view(-1, 1)).sum().item()
    end_time = time.time()
    total_time = end_time - start_time
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    time_per_image = (total_time / len(dataloader.dataset)) * 1000  # Convert to milliseconds

    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
        print('Total time: {:.4f} seconds, Time per image: {:.4f} ms'.format(
            total_time, time_per_image))
    
    return average_loss, accuracy1, total_time, time_per_image

def get_masks(module):
    """Returns an iterator over module masks, yielding the mask."""
    if hasattr(module, 'weight_mask'):
        yield module.weight_mask
    if hasattr(module, 'bias_mask'):
        yield module.bias_mask

def apply_masks(model):
    """Applies the mask to the parameters."""
    for module in model.modules():
        if hasattr(module, 'weight_mask'):
            module.weight.data.mul_(module.weight_mask)
        if hasattr(module, 'bias_mask'):
            module.bias.data.mul_(module.bias_mask)

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

def run(args):
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)
    args.post_epochs=args.epochs

    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset) 
    prune_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.prune_dataset_ratio * num_classes)
    train_loader = load.dataloader(args.dataset, args.batch_size, True, args.workers)
    test_loader = load.dataloader(args.dataset, args.batch_size, False, args.workers)

    ## Model, Loss, Optimizer ##
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    model = load.model(args.model, args.model_class)(input_shape, 
                                                     num_classes, 
                                                     args.dense_classifier, 
                                                     args.pretrained, args.pretrained_path).to(device)
    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)

    ## Pre-Train ##
    print('Pre-Train for {} epochs.'.format(args.pre_epochs))
    pre_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                 test_loader, device, args.pre_epochs, args.verbose, args.result_dir)

    ## Prune ##
    if args.experiment == "multishot":
        # Iteratively prune for `args.prune_epochs` after initial pre-training
        for epoch in range(args.prune_epochs):
            print("ashish")
            print('Pruning epoch {} of {}'.format(epoch + 1, args.prune_epochs))
            pruner = load.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
            # Increase sparsity over the first `args.sparsity_increase_epochs` epochs
            #sparsity = 10**(-float(args.compression) * ((epoch + 1) / args.prune_epochs))
            sparsity=1-float(args.compression)* ((epoch + 1) / args.prune_epochs)
            prune_loop(model, loss, pruner, prune_loader, device, sparsity, 
                       args.compression_schedule, args.mask_scope, 1, args.reinitialize, args.prune_train_mode, args.shuffle, args.invert)
            
            train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                            test_loader, device, 1, args.verbose, args.result_dir)
    else:
        # Single-shot pruning
        print('Pruning with {} for {} epochs.'.format(args.pruner, args.prune_epochs))
        pruner = load.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
        #sparsity = 10**(-float(args.compression))
        sparsity=1-float(args.compression)
        prune_loop(model, loss, pruner, prune_loader, device, sparsity, 
                   args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize, args.prune_train_mode, args.shuffle, args.invert)

    apply_masks(model)
    ## Post-Train ##
    print('Post-Training for {} epochs.'.format(args.post_epochs))
    post_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                test_loader, device, args.post_epochs, args.verbose, args.result_dir) 

    ## Display Results ##
    frames = [pre_result.head(1), pre_result.tail(1), post_result.head(1), post_result.tail(1)]
    train_result = pd.concat(frames, keys=['Init.', 'Pre-Prune', 'Post-Prune', 'Final'])

    prune_result = metrics.summary(model, 
                                pruner.scores,
                                metrics.flop(model, input_shape, device),
                                lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))
    total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
    possible_params = prune_result['size'].sum()
    pruned_parameters = total_params - possible_params
    total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
    possible_flops = prune_result['flops'].sum()
    print("Train results:\n", train_result)
    print("Prune results:\n", prune_result)
    print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
    print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))

    # Extract the 'Final' row directly
    final_result = train_result.loc['Final'].iloc[-1][['train_loss', 'test_loss', 'train_accuracy1', 'test_accuracy1']]

    print("Final Training Results:\n", final_result)


    # Calculate and print the sparsity
    sparsity_dict, total_params, sparse_params, overall_sparsity = calculate_sparsity(model)
    print_sparsity(sparsity_dict, total_params, sparse_params, overall_sparsity)

    # Calculate FPS
    input_shape = (1, 3, 32, 32)  # Example input shape, adjust according to your model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Calculating FPS...")
    fps = calculate_fps(model, input_shape, device=device)
    print("Average FPS:", fps)

    # Collect model metrics
    model_metrics = {
        'total_params': total_params,
        'pruned_params': sparse_params,
        'possible_flops': possible_flops,
        'total_flops': total_flops,
        'final_result': final_result
    }

    _, _, total_time, time_per_image = evaluate_model(model, loss, test_loader, device, args.verbose)

    # Plot model metrics
    plot_model_metrics(model_metrics, fps, total_time, time_per_image, output_dir=args.output_dir)


    ## Save Results and Model ##
    if args.save:
        print('Saving results.')
        pre_result.to_pickle("{}/pre-train.pkl".format(args.result_dir))
        post_result.to_pickle("{}/post-train.pkl".format(args.result_dir))
        prune_result.to_pickle("{}/compression.pkl".format(args.result_dir))
        torch.save(model.state_dict(),"{}/model.pt".format(args.result_dir))
        torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
        torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))

def define_args():
    parser = argparse.ArgumentParser(description='Network Compression')
    
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default='mnist',
                               choices=['mnist', 'cifar10', 'cifar100', 'tiny-imagenet', 'imagenet'],
                               help='dataset (default: mnist)')
    training_args.add_argument('--model', type=str, default='fc', choices=['fc', 'conv',
                                                                           'vgg11', 'vgg11-bn', 'vgg13', 'vgg13-bn', 'vgg16', 'vgg16-bn', 'vgg19', 'vgg19-bn',
                                                                           'resnet18', 'resnet20', 'resnet32', 'resnet34', 'resnet44', 'resnet50',
                                                                           'resnet56', 'resnet101', 'resnet110', 'resnet110', 'resnet152', 'resnet1202',
                                                                           'wide-resnet18', 'wide-resnet20', 'wide-resnet32', 'wide-resnet34', 'wide-resnet44', 'wide-resnet50',
                                                                           'wide-resnet56', 'wide-resnet101', 'wide-resnet110', 'wide-resnet110', 'wide-resnet152', 'wide-resnet1202'],
                               help='model architecture (default: fc)')
    training_args.add_argument('--model-class', type=str, default='lottery', choices=['default', 'lottery', 'tinyimagenet', 'c'],
                               help='model class (default: lottery)')
    training_args.add_argument('--dense-classifier', type=bool, default=False,
                               help='ensure last layer of model is dense (default: False)')
    training_args.add_argument('--pretrained', type=bool, default=False,
                               help='load pretrained weights (default: False)')
    training_args.add_argument('--pretrained_path', type=str, default=None,
                               help='load pretrained weights from the path (default: None)')
    training_args.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'momentum', 'adam', 'rms'],
                               help='optimizer (default: adam)')
    training_args.add_argument('--train-batch-size', type=int, default=64,
                               help='input batch size for training (default: 64)')
    training_args.add_argument('--test-batch-size', type=int, default=256,
                               help='input batch size for testing (default: 256)')
    training_args.add_argument('--pre-epochs', type=int, default=0,
                               help='number of epochs to train before pruning (default: 0)')
    training_args.add_argument('--epochs', default=10, type=int, help='Number of training epochs')
    training_args.add_argument('--post_epochs', type=int, default=10,
                               help='number of epochs to train after pruning (default: 10)')
    training_args.add_argument('--lr', type=float, default=0.001,
                               help='learning rate (default: 0.001)')
    training_args.add_argument('--lr-drops', type=int, nargs='*', default=[],
                               help='list of learning rate drops (default: [])')
    training_args.add_argument('--lr-drop-rate', type=float, default=0.1,
                               help='multiplicative factor of learning rate drop (default: 0.1)')
    training_args.add_argument('--weight-decay', type=float, default=0.0,
                               help='weight decay (default: 0.0)')
    training_args.add_argument('--batch_size', type=int, default=256,
                               help='batch size')
    
    # Pruning Hyperparameters
    pruning_args = parser.add_argument_group('pruning')
    pruning_args.add_argument('--pruner', type=str, default='rand',
                              choices=['rand', 'mag', 'snip', 'grasp', 'synflow', 'our_algo'],
                              help='prune strategy (default: rand)')
    pruning_args.add_argument('--compression', type=float, default=0.0,
                              help='quotient of prunable non-zero prunable parameters before and after pruning (default: 1.0)')
    pruning_args.add_argument('--prune_epochs', type=int, default=3,
                              help='number of iterations for scoring (default: 1)')
    pruning_args.add_argument('--compression-schedule', type=str, default='exponential', choices=['linear', 'exponential'],
                              help='whether to use a linear or exponential compression schedule (default: exponential)')
    pruning_args.add_argument('--mask-scope', type=str, default='global', choices=['global', 'local'],
                              help='masking scope (global or layer) (default: global)')
    pruning_args.add_argument('--prune-dataset-ratio', type=int, default=10,
                              help='ratio of prune dataset size and number of classes (default: 10)')
    pruning_args.add_argument('--prune-batch-size', type=int, default=256,
                              help='input batch size for pruning (default: 256)')
    pruning_args.add_argument('--prune-bias', type=bool, default=False,
                              help='whether to prune bias parameters (default: False)')
    pruning_args.add_argument('--prune-batchnorm', type=bool, default=False,
                              help='whether to prune batchnorm layers (default: False)')
    pruning_args.add_argument('--prune-residual', type=bool, default=False,
                              help='whether to prune residual connections (default: False)')
    pruning_args.add_argument('--prune-train-mode', type=bool, default=False,
                              help='whether to prune in train mode (default: False)')
    pruning_args.add_argument('--reinitialize', type=bool, default=False,
                              help='whether to reinitialize weight parameters after pruning (default: False)')
    pruning_args.add_argument('--shuffle', type=bool, default=False,
                              help='whether to shuffle masks after pruning (default: False)')
    pruning_args.add_argument('--invert', type=bool, default=False,
                              help='whether to invert scores during pruning (default: False)')
    pruning_args.add_argument('--pruner-list', type=str, nargs='*', default=[],
                              help='list of pruning strategies for singleshot (default: [])')
    pruning_args.add_argument('--prune-epoch-list', type=int, nargs='*', default=[],
                              help='list of prune epochs for singleshot (default: [])')
    pruning_args.add_argument('--compression-list', type=float, nargs='*', default=[],
                              help='list of compression ratio exponents for singleshot/multishot (default: [])')
    pruning_args.add_argument('--level-list', type=int, nargs='*', default=[],
                              help='list of number of prune-train cycles (levels) for multishot (default: [])')
    pruning_args.add_argument('--sparsity-increase-epochs', type=int, default=5,
                              help='Number of epochs over which to increase sparsity dynamically')
    
    ## Experiment Hyperparameters ##
    parser.add_argument('--experiment', type=str, default='singleshot',
                        choices=['singleshot', 'multishot', 'unit-conservation',
                                 'layer-conservation', 'imp-conservation', 'schedule-conservation'],
                        help='experiment name (default: example)')
    parser.add_argument('--expid', type=str, default='',
                        help='name used to save results (default: "")')
    parser.add_argument('--output_dir', type=str, default='',
                        help='name used to save results (default: "")')
    parser.add_argument('--result-dir', type=str, default='Results/data',
                        help='path to directory to save results (default: "Results/data")')
    parser.add_argument('--gpu', type=int, default='0',
                        help='number of GPU device to use (default: 0)')
    parser.add_argument('--workers', type=int, default='4',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--verbose', action='store_true',
                        help='print statistics during training and testing')
    
    return parser


if __name__ == '__main__':
    parser = define_args()
    args = parser.parse_args()
    args.expid=args.output_dir
    ## Construct Result Directory ##
    if args.expid == "":
        print("WARNING: this experiment is not being saved.")
        setattr(args, 'save', False)
    else:
        result_dir = args.expid
        setattr(args, 'save', True)
        setattr(args, 'result_dir', result_dir)
        try:
            os.makedirs(result_dir, exist_ok=True)  # Create the directory if it does not exist, without raising an error if it does
        except Exception as e:
            print(f"An error occurred while creating the directory: {e}")

    ## Save Args ##
    if args.save:
        with open(args.result_dir + '/args.json', 'w') as f:
            json.dump(args.__dict__, f, sort_keys=True, indent=4)

    run(args)
