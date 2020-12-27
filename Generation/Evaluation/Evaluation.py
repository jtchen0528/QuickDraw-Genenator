import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import torch.utils.data as data
import pandas as pd
import argparse
from torch.autograd import Variable
from skimage import transform

from Models import DCGAN_Generator, DCCGAN_Generator


def getModel(name, img_size, latent_dim, channels, classes):
    if name == "DCGAN":
        return DCGAN_Generator(size=img_size, lat=latent_dim, channels=channels)
    if name == "DCCGAN":
        return DCCGAN_Generator(labels=classes, d=128)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick, Draw! data Evaluation for final project.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--models_root', '-r', type=str, default='Models',
                        help='root for the dataset directory.')
    parser.add_argument('--model', '-m', type=str,
                        default='DCGAN', help='choose the model.')
    parser.add_argument("--output_num", '-num', type=int, default=10,
                        help="number of output image")
    parser.add_argument("--output_dir", '-out_dir', type=str, default="output",
                        help="directory of output csv")
    parser.add_argument('--eval_bs', '-eb', type=int,
                        default=64, help='evaluation batch size.')
    parser.add_argument("--latent_dim", '-lat', type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", '-img', type=int, default=32,
                        help="size of each image dimension (size * size)")
    parser.add_argument("--channels", '-ch', type=int, default=1,
                        help="number of image channels")
    parser.add_argument("--classes", '-c', type=int, default=10,
                        help="number of classes")
    parser.add_argument("--gen_img", '-gen', type=int, default=1,
                        help="enable/disable image generation")

    args = parser.parse_args()

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    os.makedirs(args.output_dir + '/csv', exist_ok=True)

    files = os.listdir("./" + args.models_root)
    for f in files:

        generator = getModel(args.model, args.img_size, args.latent_dim, args.channels, args.classes)
        generator.load_state_dict(torch.load(
            './' + args.models_root + '/' + f))
        if cuda:
            generator.cuda()

        label = f.split("_")[1]
        print("Model loaded, start evaluating " + label + ".")

        if args.model == "DCGAN": 

            z = Variable(Tensor(np.random.normal(
                -1, 1, (args.img_size, args.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)[:args.output_num]

            gen_imgs = gen_imgs.reshape(
                args.output_num, args.img_size, args.img_size)

            gen_imgs = gen_imgs.cpu().detach().numpy()

            print(gen_imgs.shape)

            del generator
            i = 0
            for img in gen_imgs:
                img = transform.resize(img, (64, 64))
                i = i + 1
                np.savetxt("./" + args.output_dir + "/csv/" + label + "_" +
                        str(i) + ".csv", img, delimiter=",")

            print("Csv generation complete.")

            if args.gen_img :
                os.makedirs(args.output_dir + '/images', exist_ok=True)

                files = os.listdir("./" + args.output_dir + '/csv')
                for f in files:
                    img = np.genfromtxt('./' + args.output_dir + '/csv/' + f, delimiter=',')
                    plt.imshow(img)
                    plt.savefig('./' + args.output_dir + '/images/' + f[:-4] + '.png')

        if args.model == "DCCGAN":
                # fixed noise & label
            classes = args.classes

            onehot = torch.zeros(classes, classes)
            onehot = onehot.scatter_(1, torch.LongTensor(
                np.arange(classes)).view(classes, 1), 1).view(classes, classes, 1, 1)

            z_ = torch.randn((args.output_num * classes, classes * classes)
                             ).view(-1, classes * classes, 1, 1)

            fixed_y_ = torch.zeros(args.output_num, 1)          
            for i in range(classes - 1):
                temp = torch.ones(args.output_num, 1) + i
                fixed_y_ = torch.cat([fixed_y_, temp], 0)

            y_ = (fixed_y_).type(torch.LongTensor).squeeze()
            y_label_ = onehot[y_]
            z_, y_label_ = Variable(z_.cuda()), Variable(
                y_label_.cuda())

            generator.eval()
            gen_imgs = generator(z_, y_label_)
            gen_imgs = Variable(gen_imgs.cpu())

            del generator
            i = 0
            for img in gen_imgs:
                i = i + 1
                np.savetxt("./" + args.output_dir + "/csv/" + label + "_" +
                        str(i) + ".csv", img[0], delimiter=",")

            print("Csv generation complete.")

            if args.gen_img :
                os.makedirs(args.output_dir + '/images', exist_ok=True)

                files = os.listdir("./" + args.output_dir + '/csv')
                for f in files:
                    img = np.genfromtxt('./' + args.output_dir + '/csv/' + f, delimiter=',')
                    plt.imshow(img)
                    plt.savefig('./' + args.output_dir + '/images/' + f[:-4] + '.png')
