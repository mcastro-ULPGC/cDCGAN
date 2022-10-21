import numpy as np
import torch.nn as nn
from config import *
import torch.optim as optim
from collections import OrderedDict


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# GENERATOR CODE
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        n_hidden_layers = np.log2(image_size).astype(int) - 3 # image_size >= 2**2
        layers = []
        keys = [] 
        for i in reversed(range(n_hidden_layers)):
            layers.append(nn.ConvTranspose2d(ngf * 2 ** (i + 1), ngf * 2 ** i, 4, 2, 1, bias=False))
#            layers.append(nn.Dropout(0.5)) #TIP (not sure of including it)
            layers.append(nn.BatchNorm2d(ngf * 2 ** i))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            keys.append('convtr2d'+str(n_hidden_layers - i))
            keys.append('bn'+str(n_hidden_layers - i))
            keys.append('lrelu'+str(n_hidden_layers - i))
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
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        n_hidden_layers = np.log2(image_size).astype(int) - 3 # image_size >= 2**2
        layers = []
        keys = [] 
        for i in range(n_hidden_layers):
            layers.append(nn.Conv2d(ndf * 2 ** i, ndf * 2 ** (i + 1), 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(ndf * 2 ** (i + 1)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            keys.append('conv'+str(i))
            keys.append('bn'+str(i))
            keys.append('lrelu'+str(i))
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

def set_GAN():
    netG = set_Generator(ngpu,device)
    netD = set_Discriminator(ngpu,device)
    return netG, netD

def set_GAN_loss():
    # Initialize BCELoss function
    criterion = nn.BCELoss()
    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(image_size, nz, 1, 1, device=device) # image_size = 64 ??
    # Establish convention for real and fake labels during training
    # NEW: SOFT LABELS
    real_label = 0.75 + torch.randn(1,device=device) * 0.5
    fake_label = 0.00 + torch.randn(1,device=device) * 0.3
    return criterion, fixed_noise, real_label, fake_label

def set_GAN_optimizer(netG,netD):
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
#    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.SGD(netG.parameters(), lr=lr, momentum=0.9)
    return optimizerD, optimizerG

def set_GAN_scheduler(optimizerD,optimizerG):
    # Setup Adam optimizers for both G and D
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=1, gamma=0.5)
    schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=1, gamma=0.5)
    return schedulerD, schedulerG


