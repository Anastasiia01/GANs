from __future__ import print_function
import sys
import argparse
import os
import random
import torch
import torch.nn as nn
#import torch.nn.parallel
#import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from IPython.display import HTML
from Discriminator import Discriminator
from Generator import Generator
from Utils import Utils
from TrainDG import TrainDG
import matplotlib.image as mimg

# Reference: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # to produce different results each time
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# -------------define constants and parameters--------------------
# Root directory for dataset
dataroot = "./data2"
workers = 0  # number of workers for dataloader, 2 creates problems
batch_size = 128  # training batch_size
image_size = 64 # all images in dataset resized to this size
nc = 3 # num channels in the training images, 3 for color
nz = 100 # size of z latent vector (i.e. size of generator input)
ngf = 64 # size of feature maps in generator
ndf = 64 # size of feature maps in discriminator

num_epochs = 30 # number of training epochs
lr = 0.0002 # learning rate for optimizers
beta1 = 0.5 # Beta1 hyperparam for Adam optimizers

ngpu = 1 # number of GPUs available. Use 0 for CPU mode.
#--------------------------end constants and parameters---------------

# device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#---------------weight initialization--------------
# custom weights initialization called on netG and netD
# initialize all weights with a mean of 0 and std dev of 0.02
def weights_init(m):
    classname = m.__class__.__name__
    #print("classname",classname)
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
#--------------------------------------------------

def main():
    #--------------prepare data------------------------
    utils = Utils()
    dataloader, test_dataloader = utils.prepare_data(dataroot, image_size, batch_size, workers)
    utils.plot_training_images(dataloader, device)


    # ---------------Create the generator---------------
    netG = Generator(ngpu, nz,nc, ngf).to(device)
    # handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    print(netG) # print the generator model
    #-----------------end Generator-------------------

    #------------------create discriminator-----------------
    netD = Discriminator(ngpu, nc, ndf).to(device)
    # handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)
    # Print the model
    print(netD)
    #-------------------------------------------------------

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    #----------------end BCE--------------------------------

    #-------------------Training Loop-----------------------
    # Training Loop
    trainDG = TrainDG()
    img_list, G_losses, D_losses = trainDG.train_disc_gen(dataloader, nz, netD, netG, optimizerD, optimizerG, num_epochs, device)

    #----------------results---------------------------------
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    #plt.savefig('./results/Loss3.png')
    #--------------------------------------------------------

    #------------------------animation of G's progress-------
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800) 
    ani.save('./results/G_progress4.mp4', writer=writer)

    #HTML(ani.to_jshtml())
    #--------------------------------------------------------

    #------------real/fake images side by side---------------
    # some real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
    plt.savefig('./results/real_img4.png')
    # ---------Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    mimg.imsave('./results/fake_imgs4.png', np.transpose(img_list[-1].numpy(),(1,2,0)))
    plt.show()
    #------------------------end real/fake side by side--------------


if __name__ == "__main__":
    sys.exit(int(main() or 0))