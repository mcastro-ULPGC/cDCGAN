from genericpath import exists
import numpy as np
import torch.nn as nn
from config import *
import torch.optim as optim
from collections import OrderedDict

# Decide which device we want to run on
import torch
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class GaussianNoise(nn.Module):
    def __init__(self, mean=0., std=1.):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return x + torch.normal(mean=self.mean, std=self.std, size=x.size(),device=device)

# GENERATOR CODE
class Generator(nn.Module):
    def __init__(self, ngpu, drop=0.5, noise=0):
        super(Generator, self).__init__()
        n_hidden_layers = np.log2(image_size).astype(int) - 3 # image_size >= 2**2
        layers = []
        keys = [] 
        for i in reversed(range(n_hidden_layers)):
            layers.append(GaussianNoise(std=noise)) if noise !=0 else None            
            layers.append(nn.ConvTranspose2d(ngf * 2 ** (i + 1), ngf * 2 ** i, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(ngf * 2 ** i))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            layers.append(nn.Dropout(drop)) if drop !=0 else None 
            keys.append('noise'+str(i)) if noise !=0 else None
            keys.append('convtr2d'+str(n_hidden_layers - i))
            keys.append('bn'+str(n_hidden_layers - i))
            keys.append('lrelu'+str(n_hidden_layers - i))
            keys.append('dropout'+str(n_hidden_layers - i)) if drop !=0 else None
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # first layer
            nn.ConvTranspose2d(nz, ngf * 2 ** n_hidden_layers, 4, 1, bias=False),
            nn.BatchNorm2d(ngf * 2 ** n_hidden_layers),
            nn.LeakyReLU(0.2,inplace=True),
            # hidden layers
            nn.Sequential(OrderedDict(dict(zip(keys, layers)))),
            # last layer
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)

def set_Generator(ngpu,device,show=False):
    # Create the generator
    netG = Generator(ngpu).to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02.
    netG.apply(weights_init)
    # Print the model
    print(netG) if show else None
    return netG

# DISCRIMINATOR CODE
class Discriminator(nn.Module):
    def __init__(self, ngpu,drop=0):
        super(Discriminator, self).__init__()
        n_hidden_layers = np.log2(image_size).astype(int) - 3 # image_size >= 2**2
        layers = []
        keys = [] 
        for i in range(n_hidden_layers):
            layers.append(nn.Conv2d(ndf * 2 ** i, ndf * 2 ** (i + 1), 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(ndf * 2 ** (i + 1)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(drop)) if drop !=0 else None
            keys.append('conv'+str(i))
            keys.append('bn'+str(i))
            keys.append('lrelu'+str(i))
            keys.append('dropout'+str(i)) if drop !=0 else None
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # first layer
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # hidden layers
            nn.Sequential(OrderedDict(dict(zip(keys, layers)))),
            # last layer
            nn.Conv2d(ndf * 2 ** (i + 1), 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)

def set_Discriminator(ngpu,device,show=False):
    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02.
    netD.apply(weights_init)
    # Print the model
    print(netD) if show else None
    return netD

def set_GAN(show=False):
    netD = set_Discriminator(ngpu,device,show=show)
    netG = set_Generator(ngpu,device,show=show)
    return netD,netG

def set_GAN_loss():
    from config import label_softness
    import random
    # Initialize BCELoss function
    criterion = nn.BCELoss()
    # Create batch of latent vectors that we will use to visualize the progression of the generator
#    fixed_noise = torch.normal(mean=0,std=1,size=(image_size, nz, 1, 1),device=device)
#    fixed_noise = torch.normal(mean=0.5,std=0.5,size=(image_size, nz, 1, 1),device=device)
    fixed_noise = torch.randn(image_size, nz, 1, 1, device=device)
    # Establish convention for real and fake labels during training
    real_label = - label_softness * random.normalvariate(0.5, 0.1) + 1.00
    fake_label = + label_softness * random.normalvariate(0.5, 0.1)
    return criterion, fixed_noise, real_label, fake_label

def set_GAN_optimizer(netD,netG):
    optimizerD = optim.SGD(netD.parameters(), lr=lr_D, momentum=momentum, nesterov=nesterov)
    optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))
    return optimizerD, optimizerG

def set_GAN_scheduler(optimizerD,optimizerG):
    # Setup Adam optimizers for both G and D
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=scheduler_step_size_D, gamma=scheduler_gamma_D)
    schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=scheduler_step_size_G, gamma=scheduler_gamma_G)
    return schedulerD, schedulerG

def set_GAN_train():
    netD,netG = set_GAN()
    criterion, fixed_noise, real_label, fake_label = set_GAN_loss()
    optimizerD, optimizerG = set_GAN_optimizer(netD,netG)
    schedulerD, schedulerG = set_GAN_scheduler(optimizerD,optimizerG)
    return netD,netG, criterion, fixed_noise, real_label, fake_label, optimizerD, optimizerG, schedulerD, schedulerG

