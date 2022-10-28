import torchvision.utils as vutils
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from init import *
from config import *
from output import *

def switch_labels(label,real_label,fake_label):
    if label == real_label:
        label = fake_label
    else:
        label = real_label
    return label

def add_noise(image,noisy_input=0.5,mean=0,std=1):
    return image + noisy_input*torch.normal(mean=mean, std=std, size=(image.shape), device=device)
    

def train_D(netD,netG,optimizerD,criterion,label,fake_label,real_cpu,b_size):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    netD.zero_grad()
    # Forward pass real batch through D
    real_image = add_noise(real_cpu) if noisy_input else real_cpu
    output = netD(real_image).view(-1)
    # Calculate loss on all-real batch
    errD_real = criterion(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()
    ## Train with all-fake batch
    # Generate batch of latent vectors
    noise = torch.randn(b_size, nz, 1, 1, device=device)
    # Generate fake image batch with G
    fake = netG(noise)
    label.fill_(fake_label)
    # Classify all fake batch with D
    fake_image = add_noise(fake) if noisy_input else fake
    output = netD(fake_image.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    errD_fake = criterion(output, label)
    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    errD_fake.backward()
    D_G_z1 = output.mean().item()
#    clip_grad_norm_(parameters=netD.parameters(), max_norm=8, norm_type=0.5) # CHECK
    clip_grad_value_(parameters=netD.parameters(), clip_value=0.1) # WGAN
    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake
    # Update D
    optimizerD.step()
    return fake_image, errD_real, errD_fake, D_x, D_G_z1

def train_G(fake,netD,netG,optimizerG,criterion,real_label,b_size):
    torch.autograd.set_detect_anomaly(True)
    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
    label.fill_(real_label)  # fake labels are real for generator cost
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = netD(fake).view(-1)
    # Calculate G's loss based on this output
    errG = criterion(output, label)
    # Calculate gradients for G
    errG.backward()
    D_G_z2 = output.mean().item()
    # Update G
    optimizerG.step()
    return errG, D_G_z2

def train_GAN(dataloader,netD,netG,optimizerD,optimizerG,criterion,real_label,fake_label,fixed_noise,schedulerD=None, schedulerG=None):
    # Lists to keep track of progress
    img_list, D_real_losses, D_fake_losses, G_losses, D_real, D_fake, G_acc = [], [], [], [], [], [], []
    iters, n_fake = 0
    print_start()
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Update D network
            fake,errD_real,errD_fake,D_x,D_G_z1 = train_D(netD,netG,optimizerD,criterion,label,fake_label,real_cpu,b_size)
            # Update G network
            errG, D_G_z2 = train_G(fake,netD,netG,optimizerG,criterion,real_label,b_size)
            # Output training stats
            print_training_progress(iters,epoch,i,len(dataloader),errD_real,errD_fake,errG,D_x,D_G_z1,D_G_z2)
            # Save Losses and Gradients for plotting later
            D_real_losses.append(errD_real.item())
            D_fake_losses.append(errD_fake.item())
            G_losses.append(errG.item())
            D_real.append(D_x)
            D_fake.append(D_G_z1)
            G_acc.append(D_G_z2)
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    save_image(fake, n_fake)
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            # Next iteration
            iters += 1
        # Next epoch
        if schedulerD is not None:
            schedulerD.step()
        if schedulerG is not None:
            schedulerG.step()
    return D_real_losses, D_fake_losses, G_losses, img_list, D_real, D_fake, G_acc

