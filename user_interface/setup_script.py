import os
import subprocess
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

print("ashish")

# Ensure the Data directory exists and change to it
data_dir = "../Data"
os.makedirs(data_dir, exist_ok=True)
os.chdir(data_dir)

# Install required packages and download data if not already present
if not os.path.exists('open-images-bus-trucks'):
    subprocess.run(['wget', '--quiet', 'https://www.dropbox.com/s/agmzwk95v96ihic/open-images-bus-trucks.tar.xz'])
    subprocess.run(['tar', '-xf', 'open-images-bus-trucks.tar.xz'])
    os.remove('open-images-bus-trucks.tar.xz')


# Move to the 'annotations' directory and copy annotation files
os.chdir("open-images-bus-trucks/annotations")
subprocess.run(['cp', 'mini_open_images_train_coco_format.json', 'instances_train2017.json'])
subprocess.run(['cp', 'mini_open_images_val_coco_format.json', 'instances_val2017.json'])

# Move back to the Experiments_FPGM directory
os.chdir("../../")

# Change back to the Data directory
os.chdir("open-images-bus-trucks")

# Create symbolic links for image directories
if not os.path.exists('train2017'):
    os.symlink('images/', 'train2017')
if not os.path.exists('val2017'):
    os.symlink('images/', 'val2017')


# Define the classes for detection
CLASSES = ['', 'BUS', 'TRUCK']
os.chdir("../")
# Import required libraries
from torch_snippets import *

# Check if the pretrained model is available, and if not, download it
if not os.path.exists('detr-r50-e632da11.pth'):
    subprocess.run(['wget', 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'])
    checkpoint = torch.load("detr-r50-e632da11.pth", map_location='cpu')
    del checkpoint["model"]["class_embed.weight"]
    del checkpoint["model"]["class_embed.bias"]
    torch.save(checkpoint, "detr-r50_no-class-head.pth")

# Import necessary libraries for CIFAR-10 dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Download and transform CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../Data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../Data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
