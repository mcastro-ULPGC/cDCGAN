############################ CONFIGURATIONS #############################
"""
This file contains all the configurations for the project.
"""
# Number of workers for dataloader
workers = 1

############## Input Data ###############################################

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Root directory for dataset
dataroot = r"C:\Users\mcastro\Documents\MCastro\2_Codigo\0_DATASETS\SkinCancerBvsM\data"
#dataroot = r"C:\Users\mcastro\Documents\MCastro\2_Codigo\0_DATASETS\PH2\Clases"
#dataroot = r"C:\Users\mcastro\Documents\MCastro\2_Codigo\0_DATASETS\HAM10000_todo\Clases"

# Batch size during training
batch_size = 256

# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 256

# Size of feature maps in generator
ngf = 128

# Size of feature maps in discriminator
ndf = 128

# Making the labels soft (if 0, then the labels are hard)
label_softness = 0.15

# Number of training epochs
num_epochs = 10

############## Adjusting D hyperparameters ##############################
# Adding noise to the discriminator's input
noisy_input = 1

# Learning rate for optimizers
lr_D = 0.0003

# Momentum hyperparam for SGD optimizer (D)
momentum = 0.9
nesterov = True

# Scheduler
scheduler_step_size_D = 2
scheduler_gamma_D = 0.1

############## Adjusting G hyperparameters ##############################
# Learning rate for optimizers
lr_G = 0.0003

# Beta1 hyperparam for Adam optimizer (G)
beta1 = 0.9

# Scheduler
scheduler_step_size_G = 2
scheduler_gamma_G = 0.1

############## Output Data ##############################################



#########################################################################