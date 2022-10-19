from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.utils.data

# Root directory for dataset
#dataroot = "data/celeba"
dataroot = r"C:\Users\mcastro\Documents\MCastro\2_Codigo\0_DATASETS\SkinCancerBvsM\data"

# Number of workers for dataloader
workers = 1

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 500

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# ___
model_dir = os.getcwd()
model_name = 't0001'

######################################################################
# Decide which device we want to run on
import torch
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)