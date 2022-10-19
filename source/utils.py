from config import *
import torchvision.utils as vutils
import matplotlib.pyplot as plt

def train_GAN(dataloader,netG,netD,optimizerD,optimizerG,criterion,real_label,fake_label,fixed_noise):
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
#            print(output,output.shape)
#            print(label,label.shape)
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
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
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
            ############################
            # Output training stats
            message = '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, num_epochs, i, len(dataloader),errD.item(),errG.item(),D_x,D_G_z1,D_G_z2)
            if i % 50 == 0:
                print(message)
                with open('results\log.txt', 'a') as f:
                    f.write(message)
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            # Next iteration
            iters += 1
    return G_losses, D_losses, img_list

def plot_loss(G_losses,D_losses):
    """ Plot the training losses for the generator and discriminator, recorded during training """
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_G_progress(img_list):
    """ Visualization of G’s progression """
    import numpy as np
    import matplotlib.animation as animation
    from IPython.display import HTML
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())

def plot_Real_VS_Fake(dataloader,img_list):
    """ Plot some real images and fake images from the last epoch """
    import numpy as np
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show() 

def save_model(netG, netD, G_losses, D_losses, img_list, model_dir, model_name):
    import numpy as np
    # Save the model
    torch.save(netG.state_dict(), model_dir + 'models' + model_name + '_generator.pth')
    torch.save(netD.state_dict(), model_dir + 'models' + model_name + '_discriminator.pth')
    # Save the losses
#    np.save(model_dir + 'results' + model_name + '_G_losses.npy', G_losses)
#    np.save(model_dir + 'results' + model_name + '_D_losses.npy', D_losses)
    # Save the images
#    np.save(model_dir + 'results' + model_name + '_img_list.npy', img_list)

