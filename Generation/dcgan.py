import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import math

from models import DCGAN_Generator, DCGAN_Discriminator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Convolutional GAN for Quick Draw doodle generation.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--object", '-o', type=str, default='airplane',
                        help="object for generator to learn")
    parser.add_argument("--data_root", '-r', type=str, default='Data',
                        help="training data root")
    parser.add_argument("--epochs", '-e', type=int, default=20,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", '-bs', type=int,
                        default=128, help="size of the batches")
    parser.add_argument("--lr", '-lr', type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", '-b1', type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", '-b2', type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", '-n_cpu', type=int, default=8,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", '-lat', type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", '-img', type=int, default=32,
                        help="size of each image dimension (size * size)")
    parser.add_argument("--channels", '-ch', type=int, default=1,
                        help="number of image channels")
    parser.add_argument("--sample_interval", '-sample', type=int,
                        default=200, help="interval between image sampling.")
    parser.add_argument("--logging", '-log', type=int,
                        default=0, help="enable/disable logging")
    parser.add_argument("--save_model", '-save_model', type=int,
                        default=1, help="save/discard trained model.")
    args = parser.parse_args()
    print(args)

    cuda = True if torch.cuda.is_available() else False

    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    class QuickDrawDataset(Dataset):
        """Quick Draw dataset."""

        def __init__(self, root, label, transform=None):
            data = np.load(f'{root}/{label}.npy')
            data = data/255
            img_w, img_h = int(math.sqrt(data.shape[1])), int(
                math.sqrt(data.shape[1]))
            data = np.reshape(data, [data.shape[0], 1, img_w, img_h])
            padding = int((args.img_size - img_w) / 2)
            self.data_frame = np.pad(data, ((0, 0), (0, 0), (padding, padding),
                                            (padding, padding)), 'constant', constant_values=(0, 0))

        def __len__(self):
            return len(self.data_frame)

        def __getitem__(self, idx):
            imgs = self.data_frame[idx]
            return imgs

    def train(epochs=200, name='object', interval=200, logging=False, save_model=True):

        if logging:
            os.makedirs("images", exist_ok=True)
            os.makedirs("images/" + name, exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            os.makedirs("logs/" + name, exist_ok=True)
            G_LOSS = []
            D_LOSS = []

        if save_model:
            os.makedirs("models", exist_ok=True)
            os.makedirs("models/" + name, exist_ok=True)

        for epoch in range(epochs):
            for i, imgs in enumerate(dataloader):
                # print(imgs.shape)
                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(
                    1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(
                    0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(
                    0, 1, (imgs.shape[0], args.latent_dim))))

                # Generate a batch of images
                gen_imgs = generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(
                    discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

            if logging:
                G_LOSS.append(float(g_loss.cpu().detach().numpy().astype(float)))
                D_LOSS.append(float(d_loss.cpu().detach().numpy().astype(float)))
                save_image(gen_imgs.data[:16], "images/" + name + "/%d.png" %
                        epoch, nrow=4, normalize=True)
                with open("./logs/" + name + "/g_loss_" + name + ".txt", "w") as output:
                    output.write(str(G_LOSS))
                with open("./logs/" + name + "/d_loss_" + name + ".txt", "w") as output:
                    output.write(str(D_LOSS))
            if save_model:
                torch.save(generator.state_dict(), './models/' + name + '/model_' + name + '_' + str(epoch) + '.pytorch')


    # load training data
    data = QuickDrawDataset(root=args.data_root, label=args.object, transform=transforms.Compose([
                            transforms.Resize(args.img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = DCGAN_Generator(size=args.img_size, lat=args.latent_dim, channels=args.channels)
    discriminator = DCGAN_Discriminator(size=args.img_size, channels=args.channels)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    train(epochs=args.epochs, name=args.object, interval=args.sample_interval, logging=args.logging, save_model=args.save_model)

    print("training " + args.object + " completed.")