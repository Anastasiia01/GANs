import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from logger import Logger
#from torch.utils.tensorboard import SummaryWriter
from Discriminator import DenseDiscriminator
from Generator import DenseGenerator


class GAN(object):
    def __init__(self, _epochs, _batch_size):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Generator architecture
        self.G = DenseGenerator().to(self.device)

        # Discriminator architecture
        self.D = DenseDiscriminator().to(self.device)

        # Binary cross entropy loss and optimizer
        self.loss = nn.BCELoss()
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, weight_decay=0.00001)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002, weight_decay=0.00001)

        # Set the logger
        self.logger = Logger('./logs')
        self.number_of_images = 10
        self.epochs = _epochs
        self.batch_size = _batch_size

    
    def train(self, train_loader, resume_training = False): 
        self.t_begin = time.time()
        generator_iter = 0

        if resume_training:
            try:
                self.load_model()
            except Exception as e:
                print(e)
                print("Failed to load model. Training from scratch")
        else:
            print("Training from scratch")

        for epoch in range(self.epochs+1):
            for i, (images, _) in enumerate(train_loader):
                # Check if round number of batches
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break
                #print(images.size())
                # Flatten image 1,32x32 to 1024
                images = images.view(self.batch_size, -1) #get images
                z = torch.rand((self.batch_size, 100)) #get z

                real_labels = Variable(torch.ones(self.batch_size, device = self.device))
                fake_labels = Variable(torch.zeros(self.batch_size, device = self.device))
                images, z = Variable(images), Variable(z)
                images = images.to(self.device)
                z = z.to(self.device)

                # Train discriminator
                # compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
                # [Training discriminator = Maximizing discriminator being correct]
                #print("Before outputs", images.size())
                outputs = self.D(images).view(-1)
                #print("Outputs:", outputs.size())
                #print("Real_labels:", real_labels.size())
                d_loss_real = self.loss(outputs, real_labels)
                real_score = outputs

                # Compute BCELoss using fake images
                fake_images = self.G(z)
                outputs = self.D(fake_images).view(-1)
                d_loss_fake = self.loss(outputs, fake_labels)
                fake_score = outputs

                # Optimizie discriminator
                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train generator
                z = Variable(torch.randn(self.batch_size, 100)).to(self.device)
                fake_images = self.G(z)
                outputs = self.D(fake_images).view(-1)

                # We train G to maximize log(D(G(z))[maximize likelihood of discriminator being wrong] instead of
                # minimizing log(1-D(G(z)))[minizing likelihood of discriminator being correct]
                # From paper  [https://arxiv.org/pdf/1406.2661.pdf]
                g_loss = self.loss(outputs, real_labels)

                # Optimize generator
                self.D.zero_grad()
                self.G.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
                generator_iter += 1


                if ((i + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (i + 1), train_loader.dataset.__len__() // self.batch_size, d_loss.data, g_loss.data))

                    z = Variable(torch.randn(self.batch_size, 100)).to(self.device)

                    # ============ TensorBoard logging ============#
                    # (1) Log the scalar values
                    info = {
                        'd_loss': d_loss.data,
                        'g_loss': g_loss.data
                    }

                    for tag, value in info.items():
                        self.logger.log_loss(tag, value, i + 1)
                        #self.logger.scalar_summary(tag, value, i + 1)

                if generator_iter % 1000 == 0:
                    print("Imag size", images.size())
                    print('Generator iter-{}'.format(generator_iter))
                    self.save_model()

                    if not os.path.exists('training_result_images/'):
                        os.makedirs('training_result_images/')
                    

                    """images_display = images.view(64, 1, 32, 32)
                    grid = vutils.make_grid(images_display, normalize=True).cpu()
                    vutils.save_image(grid, 'training_result_images/real_image_iter_{}.png'.format( 
                      str(generator_iter).zfill(3)))"""

                    # Denormalize images and save them in grid 8x8
                    z = Variable(torch.randn(self.batch_size, 100)).to(self.device)
                    samples = self.G(z)
                    samples = samples.mul(0.5).add(0.5)
                    samples = samples.data.cpu()
                    print("Sampl size", samples.size())                  
                    samples = samples.view(self.batch_size, 1, 32, 32)
                    print("Sampl size2", samples.size())
                    grid = vutils.make_grid(samples)
                    image_path = 'training_result_images/gan_image_iter_{}.png'.format( 
                      str(generator_iter).zfill(3))
                    vutils.save_image(grid, image_path)
                            
        # Plot the real images
        plt.figure(figsize=(8,8))
        plt.axis("off")
        #plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(images.to(self.device), normalize=True).cpu(),(1,2,0)))
        plt.savefig('training_result_images/real_img.png')

        # ---------Plot the fake images from the last epoch
        z = Variable(torch.randn(self.batch_size, 100)).to(self.device)
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        print("Sampl size", samples.size())                  
        samples = samples.view(self.batch_size, 1, 32, 32)
        print("Sampl size2", samples.size())
        grid = vutils.make_grid(samples)
        image_path = 'training_result_images/gan_img.png'
        vutils.save_image(grid, image_path)

        self.t_end = time.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        # Save the trained parameters
        self.save_model()

    def evaluate(self, test_loader, D_model_path, G_model_path): 
        self.load_model(D_model_path, G_model_path)
        z = Variable(torch.randn(self.batch_size, 100)).to(self.device)
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'gan_model_image.png'.")
        utils.save_image(grid, 'gan_model_image.png')

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            generated_images.append(sample.reshape(32,32))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self, D_model_filename = './discriminator.pkl', G_model_filename = './generator.pkl'):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))