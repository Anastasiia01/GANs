import torch.nn as nn


class DenseGenerator(nn.Module):
    def __init__(self, ngpu=1):
        super(DenseGenerator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),# 32 x 32 = 1024
            nn.LeakyReLU(0.2),
            nn.Tanh())
        
    def forward(self, input):
        return self.main(input)

class Generator(nn.Module):
    def __init__(self, ngpu, latent_space_z = 100, channels=3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            # input is Z latent vector, going into a convolution
            nn.ConvTranspose2d(in_channels=latent_space_z, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1),
            # output of main module --> Image (Cx32x32)
            nn.Tanh())
            # state size. (nc) x 64 x 64
        
    def forward(self, input):
        return self.main(input)
