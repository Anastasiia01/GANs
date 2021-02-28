import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

class Utils(object):
    def prepare_data(self, dataroot, batch_size, workers, dataset, model_name, channels=1):

        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * channels, (0.5,) * channels),
        ])

        if(dataset=='cifar'):
            #print("cif")
            trans = [
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * channels, (0.5,) * channels),
            ]
            #print("model", model_name, "model")
            if(model_name=='GAN'):
                print("here")
                trans=[transforms.Grayscale()]+trans
            trans = transforms.Compose(trans)
            train_dataset = dset.CIFAR10(root=dataroot, train=True, download=True, transform=trans)
            test_dataset = dset.CIFAR10(root=dataroot, train=False, download=True, transform=trans)
        elif(dataset=='mnist'):
            train_dataset = dset.MNIST(root=dataroot, train=True, download=True, transform=trans)
            test_dataset = dset.MNIST(root=dataroot, train=False, download=True, transform=trans)
        elif(dataset=='fashion-mnist'):
            train_dataset = dset.MNIST(root=dataroot, train=True, download=True, transform=trans)
            test_dataset = dset.MNIST(root=dataroot, train=False, download=True, transform=trans)            
        else:
            raise Exception("Invalid dataset option.")
        # Create the dataloader
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=workers)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=workers)
        return train_dataloader, test_dataloader

    def plot_training_images(self, dataloader, device):
        # Plot some training images
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
