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
import h5py

from Models import CGAN_Discriminator, CGAN_Generator
from Datasets import QuickDrawDataset_h5py

os.makedirs("images", exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Convolutional GAN for Quick Draw doodle generation.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--n_classes", '-c', type=int, default=10,
                        help="number of classes for dataset")
    parser.add_argument("--n_samples", '-s', type=int, default=20000,
                        help="number of samples per class")
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
    parser.add_argument("--img_size", '-img', type=int, default=64,
                        help="size of each image dimension (size * size)")
    parser.add_argument("--channels", '-ch', type=int, default=1,
                        help="number of image channels")
    parser.add_argument("--sample_interval", '-i', type=int,
                        default=200, help="interval between image sampling.")
    parser.add_argument("--logging", '-log', type=int,
                        default=1, help="enable/disable logging")
    parser.add_argument("--save_model", '-m', type=int,
                        default=1, help="save/discard trained model.")
    opt = parser.parse_args()

    print(opt)

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    cuda = True if torch.cuda.is_available() else False

    # Loss functions
    adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    generator = CGAN_Generator(opt.n_classes, img_shape, opt.latent_dim)
    discriminator = CGAN_Discriminator(opt.n_classes, img_shape)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    data = QuickDrawDataset_h5py(root=opt.data_root, img_size=opt.img_size, classes=opt.n_classes, samples=opt.n_samples, transform=transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    dataloader = DataLoader(data, batch_size=opt.batch_size, shuffle=False)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    def sample_image(n_row, batches_done, root, labels):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(
            0, 1, (n_row ** 2, opt.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        gen_imgs = generator(z, labels[:n_row * n_row])
        save_image(gen_imgs.data, root + "/%d.png" %
                   batches_done, nrow=n_row, normalize=True)

    # ----------
    #  Training
    # ----------

    def train(epochs=200, interval=200, logging=False, save_model=True, classes=30, batch_size=64, latent_dim=100):

        if logging:
            os.makedirs("images", exist_ok=True)
            os.makedirs("images_real", exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            G_LOSS = []
            D_LOSS = []

        if save_model:
            os.makedirs("models", exist_ok=True)

        for epoch in range(epochs):
            for i, (imgs, labels) in enumerate(dataloader):

                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).fill_(
                    1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(
                    0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(FloatTensor))
                labels = Variable(labels.type(LongTensor))

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(FloatTensor(np.random.normal(
                    0, 1, (batch_size, latent_dim))))
                gen_labels = Variable(LongTensor(
                    np.random.randint(0, classes, batch_size)))

                # Generate a batch of images
                gen_imgs = generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity = discriminator(gen_imgs, gen_labels)
                g_loss = adversarial_loss(validity, valid)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Loss for real images
                validity_real = discriminator(real_imgs, labels)
                d_real_loss = adversarial_loss(validity_real, valid)

                # Loss for fake images
                validity_fake = discriminator(gen_imgs.detach(), gen_labels)
                d_fake_loss = adversarial_loss(validity_fake, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                batches_done = epoch * len(dataloader) + i
                if batches_done % interval == 0:
                    if logging:
                        G_LOSS.append(
                            float(g_loss.cpu().detach().numpy().astype(float)))
                        D_LOSS.append(
                            float(d_loss.cpu().detach().numpy().astype(float)))
                        sample_image(
                            n_row=classes, batches_done=batches_done, root="images", labels=labels)
                        save_image(real_imgs[:classes * classes].data, "images_real/%d.png" %
                                   batches_done, nrow=classes, normalize=True)
                        with open("./logs/g_loss.txt", "w") as output:
                            output.write(str(G_LOSS))
                        with open("./logs/d_loss.txt", "w") as output:
                            output.write(str(D_LOSS))
                    if save_model:
                        torch.save(generator.state_dict(
                        ), './models/model_' + str(batches_done) + '.pytorch')

    train(epochs=opt.epochs, interval=opt.sample_interval, logging=opt.logging, save_model=opt.save_model,
          classes=opt.n_classes, batch_size=opt.batch_size, latent_dim=opt.latent_dim)
