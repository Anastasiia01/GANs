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

from Utils import Utils
from config import parse_args

from models.gan import GAN
#from models.dcgan import DCGAN_MODEL
#from models.wgan_clipping import WGAN_CP
#from models.wgan_gradient_penalty import WGAN_GP

# Reference: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # to produce different results each time
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)



def main(args):
    #--------------prepare data------------------------
    dataset = args.dataset
    dataroot = args.dataroot
    if not os.path.exists(dataroot):
        os.makedirs(dataroot)
    batch_size = args.batch_size 
    epochs = args.epochs
    channels = args.channels
    model = None
    model_name = args.model
    if args.model == 'GAN':
        model = GAN(epochs, batch_size)
    elif args.model == 'DCGAN':
        model = DCGAN_MODEL(args)
    elif args.model == 'WGAN-CP':
        model = WGAN_CP(args)
    elif args.model == 'WGAN-GP':
        model = WGAN_GP(args)
    else:
        print("Model type non-existing. Try again.")
        exit(-1)
    workers = 0  # number of workers for dataloader, 2 creates problems
    utils = Utils()
    train_loader, test_loader = utils.prepare_data(dataroot, batch_size, workers, dataset, model_name, channels)
    # Start model training
    resume_training = False
    if args.resume_training == 'True':
        resume_training = True
    if args.is_train == 'True':
        model.train(train_loader, resume_training)

    # start evaluating on test data
    else:
        model.evaluate(test_loader, args.load_D, args.load_G)
        # for i in range(50):
        #    model.generate_latent_walk(i)


if __name__ == "__main__":
    args = parse_args()
    sys.exit(int(main(args) or 0))