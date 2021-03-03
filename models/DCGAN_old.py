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


# -------------define constants and parameters--------------------
nc = 3 # num channels in the training images, 3 for color
nz = 100 # size of z latent vector (i.e. size of generator input)

num_epochs = 30 # number of training epochs
lr = 0.0002 # learning rate for optimizers
beta1 = 0.5 # Beta1 hyperparam for Adam optimizers

def weights_init(m):
    classname = m.__class__.__name__
    #print("classname",classname)
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
#--------------------------------------------------


# ---------------Create the generator---------------
netG = Generator(ngpu, nz,nc).to(device)
# handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

print(netG) # print the generator model
#-----------------end Generator-------------------

#------------------create discriminator-----------------
netD = Discriminator(ngpu, nc).to(device)
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