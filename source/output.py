import os
from turtle import color
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch

from init import device
from config import image_size, num_epochs
from config import *


def add_to_log(file=None,directory=None,**kwargs):
    file = 'training_log.txt' if file is None else file 
    path = os.path.join(directory,file) if directory is not None else 'results\\' + file
    with open(path, 'a') as f:
        for key, value in kwargs.items():
            f.write(key + ': ' + str(value) + '\t')
        f.write('\n')

def print_start(file=None,directory=None):
    file = 'training_log.txt' if file is None else file 
    path = os.path.join(directory,file) if directory is not None else 'results\\' + file
    with open(path, 'a') as f:
        f.write('HYPERPARAMETERS:\n')
        f.write('Root directory for dataset: ' + dataroot + '\n')
        f.write('Workers to load dataset: ' + str(workers) + '\n')
        f.write('Batch size: ' + str(batch_size) + '\n')
        f.write('Image size: ' + str(image_size) + '\n')
        f.write('Number of channels: ' + str(nc) + '\n')
        f.write('Size of z latent vector: ' + str(nz) + '\n')
        f.write('Size of feature maps in generator: ' + str(ngf) + '\n')
        f.write('Size of feature maps in discriminator: ' + str(ndf) + '\n')
        f.write('Softness of labels: ' + str(label_softness) + '\n')
        f.write('Number of epochs: ' + str(num_epochs) + '\n')
        f.write('Learning rate (D): ' + str(lr_D) + '\n')
        f.write('Beta1: ' + str(beta1) + '\n')
        f.write('Adding noise to the discriminator''s input: ' + str(noisy_input) + '\n')
        f.write('Momentum hyperparam for SGD optimizer (D): ' + str(momentum) + '\n')
        f.write('Nestrov momentum: ' + str(nesterov) + '\n')
        f.write('Scheduler''s step size (D): ' + str(scheduler_step_size_D) + '\n')
        f.write('Scheduler''s gamma (D): ' + str(scheduler_gamma_D) + '\n')
        f.write('Learning rate (G): ' + str(lr_G) + '\n')
        f.write('Scheduler''s step size (G): ' + str(scheduler_step_size_G) + '\n')
        f.write('Scheduler''s gamma (G): ' + str(scheduler_gamma_G) + '\n')
        f.write('Number of GPUs: ' + str(ngpu) + '\n')
#        f.write('Loss function: ' + loss_function + '\n')
#        f.write('lambda for gradient penalty: ' + str(lambda_gp) + '\n')
        f.write('\nStarting Training Loop...' + '\n')

def print_training_progress(iters,epoch, i, len_dataloader, errD_real, errD_fake, errG, D_x, D_G_z1, D_G_z2, file=None, directory=None):
    message = '[%3d][%3d/%3d][%2d/%2d]  Loss_D: %2.4f / %2.4f Loss_G: %2.4f\tD(x): %2.4f D(G(z)): %2.4f / %2.4f' % (iters,epoch, num_epochs-1, i, len_dataloader-1,errD_real.item(),errD_fake.item(),errG.item(),D_x,D_G_z1,D_G_z2)
    print(message)
    file = 'training_log.txt' if file is None else file
    path = os.path.join(directory,file) if directory is not None else 'results\\' + file
    with open(path, 'a') as f:
        f.write(message + '\n')

def save_image(image, path):
    ''' Save a single image '''
    vutils.save_image(image, path)

def save_fake(fake, n_fake, model_dir, model_name):
    ''' Save a generated image '''
    fake = fake.detach().cpu()
    path = os.path.join(model_dir, model_name + '_fake_samples_%03d.png' % n_fake)
    save_image(fake, path)

def plot_loss(D_real_losses, D_fake_losses, G_losses):
    import numpy as np
    ''' Plot the training losses for the generator and discriminator, recorded during training '''
    plt.figure(figsize=(10,5))
    plt.title('Generator and Discriminator Loss During Training')
#    style = 'dotted' if len(G_losses) > 100 else 'solid'
    style = 'solid'
#    style = 'dotted'
    plt.plot(D_real_losses, label='D_real', linestyle=style)
    plt.plot(D_fake_losses, label='D_fake', linestyle=style)
    plt.plot(G_losses, label='G', linestyle=style)
    min_G, max_G = float(min(np.array(G_losses))), float(max(np.array(G_losses)))
    min_D_real, max_D_real = float(min(np.array(D_real_losses))), float(max(np.array(D_real_losses)))
    min_D_fake, max_D_fake = float(min(np.array(D_fake_losses))), float(max(np.array(D_fake_losses)))
    plt.ylim(np.min([0,min_D_real,min_D_fake,min_G])*0.9,np.max([max_D_real,max_D_fake,max_G])*1.1)
    plt.xlabel('steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_acc(D_real, D_fake, G_acc):
    ''' Plot the accuracy of the generator and discriminator, recorded during training '''
    plt.figure(figsize=(10,5))
    plt.title('Generator and Discriminator Accuracy During Training')
    plt.plot(D_real,label='D_real')
    plt.plot(D_fake,label='D_fake')
    plt.plot(G_acc,label='G')
    plt.plot([0.25]*len(D_real),linestyle='dotted',linecolor='white',linewidth=0.1)
    plt.plot([0.50]*len(D_real),linestyle='dotted',linecolor='white',linewidth=0.1)
    plt.plot([0.75]*len(D_real),linestyle='dotted',linecolor='white',linewidth=0.1)
    plt.ylim([0,1.1])
    plt.yticks([0,0.25,0.5,0.75,1])
    plt.xlabel('steps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_G_progress(img_list):
    ''' Visualization of G’s progression '''
    import numpy as np
    import matplotlib.animation as animation
    from IPython.display import HTML
    fig = plt.figure(figsize=(8,8))
    plt.axis('off')
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())

def plot_Real_VS_Fake(dataloader,img_list):
    ''' Plot some real images and fake images from the last epoch '''
    import numpy as np
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.title('Real Images')
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:image_size], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.title('Fake Images')
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show() 

def save_model(netG, netD, model_dir, model_name):
    # Save the model
    torch.save(netG.state_dict(), model_dir + 'models' + model_name + '_generator.pth')
    torch.save(netD.state_dict(), model_dir + 'models' + model_name + '_discriminator.pth')


