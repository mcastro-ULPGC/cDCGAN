from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.utils.data

from config import ngpu

# Decide which device we want to run on
import torch
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def init(show=False):
    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed) if show else None
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # 
    # ___
#    model_dir = os.getcwd()
#    model_name = 't0001'