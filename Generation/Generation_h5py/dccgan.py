import os
import time
import argparse
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Models import DCCGAN_Generator, DCCGAN_Discriminator
from Datasets import QuickDrawDataset_h5py

def show_result(num_epoch, classes, show=False, save=False, path='result.png'):

    G.eval()
    test_images = G(fixed_z_, fixed_y_label_)
    G.train()

    size_figure_grid = classes
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(classes * classes):
        i = k // classes
        j = k % classes
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conditional Deep Convolutional GAN for Quick Draw doodle generation.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", '-r', type=str, default='Data',
                        help="training data root")
    parser.add_argument("--epochs", '-e', type=int, default=20,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", '-bs', type=int,
                        default=32, help="size of the batches")
    parser.add_argument("--lr", '-lr', type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--depth", '-d', type=int, default=128,
                        help="depth of model")
    parser.add_argument("--img_size", '-img', type=int, default=64,
                        help="size of each image dimension (size * size)")
    parser.add_argument("--classes", '-c', type=int, default=10,
                        help="number of classes to train")
    parser.add_argument("--samples", '-s', type=int, default=20000,
                        help="number of samples per label")
    parser.add_argument("--sample_interval", '-i', type=int,
                        default=200, help="interval between image sampling.")
    parser.add_argument("--logging", '-log', type=int,
                        default=0, help="enable/disable logging")
    parser.add_argument("--save_model", '-m', type=int,
                        default=1, help="save/discard trained model.")
    args = parser.parse_args()
    print(args)

    # training parameters
    batch_size = args.batch_size
    lr = args.lr
    train_epoch = args.epochs

    sample_size = args.samples
    img_size = args.img_size
    classes = args.classes
    depth = args.depth

    # results save folder
    data_root = args.data_root
    os.makedirs("results", exist_ok=True)
    root = 'results/'
    model = 'cDCGAN_'

    data = QuickDrawDataset_h5py(root=data_root, img_size=img_size, classes=classes, samples=sample_size, transform=transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    # fixed noise & label
    temp_z_ = torch.randn(classes, classes * classes)
    fixed_z_ = temp_z_
    fixed_y_ = torch.zeros(classes, 1)
    for i in range(classes - 1):
        fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
        temp = torch.ones(classes, 1) + i
        fixed_y_ = torch.cat([fixed_y_, temp], 0)

    fixed_z_ = fixed_z_.view(-1, classes * classes, 1, 1)
    fixed_y_label_ = torch.zeros(classes * classes, classes)
    fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
    fixed_y_label_ = fixed_y_label_.view(-1, classes, 1, 1)
    fixed_z_, fixed_y_label_ = Variable(
        fixed_z_.cuda()), Variable(fixed_y_label_.cuda())

    # network
    G = DCCGAN_Generator(classes, depth)
    D = DCCGAN_Discriminator(classes, depth)
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.cuda()
    D.cuda()

    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss()

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    if not os.path.isdir(root):
        os.mkdir(root)
    if not os.path.isdir(root + 'Fixed_results'):
        os.mkdir(root + 'Fixed_results')

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    # label preprocess
    onehot = torch.zeros(classes, classes)
    onehot = onehot.scatter_(1, torch.LongTensor(
        np.arange(classes)).view(classes, 1), 1).view(classes, classes, 1, 1)
    fill = torch.zeros([classes, classes, img_size, img_size])
    for i in range(classes):
        fill[i, i, :, :] = 1

    print('training start!')
    start_time = time.time()

    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []

        # learning rate decay
        if (epoch+1) == train_epoch - 5:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        if (epoch+1) == train_epoch - 2:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        epoch_start_time = time.time()
        y_real_ = torch.ones(batch_size)
        y_fake_ = torch.zeros(batch_size)
        y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        for i, (x_, y_) in enumerate(train_loader):
            # train discriminator D

            D.zero_grad()

            mini_batch = x_.size()[0]

            if mini_batch != batch_size:
                y_real_ = torch.ones(mini_batch)
                y_fake_ = torch.zeros(mini_batch)
                y_real_, y_fake_ = Variable(
                    y_real_.cuda()), Variable(y_fake_.cuda())

            y_ = y_.long()
            y_fill_ = fill[y_]

            x_, y_fill_ = Variable(x_.cuda()), Variable(y_fill_.cuda())

            x_ = x_.float()

            D_result = D(x_, y_fill_).squeeze()

            D_real_loss = BCE_loss(D_result, y_real_)

            z_ = torch.randn((mini_batch, classes * classes)
                             ).view(-1, classes * classes, 1, 1)
            y_ = (torch.rand(mini_batch, 1) *
                  classes).type(torch.LongTensor).squeeze()
            y_label_ = onehot[y_]
            y_fill_ = fill[y_]
            z_, y_label_, y_fill_ = Variable(z_.cuda()), Variable(
                y_label_.cuda()), Variable(y_fill_.cuda())

            G_result = G(z_, y_label_)
            D_result = D(G_result, y_fill_).squeeze()

            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_fake_score = D_result.data.mean()

            D_train_loss = D_real_loss + D_fake_loss

            D_optimizer.zero_grad()

            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.data)

            # train generator G
            G.zero_grad()

            z_ = torch.randn((mini_batch, classes * classes)
                             ).view(-1, classes * classes, 1, 1)
            y_ = (torch.rand(mini_batch, 1) *
                  classes).type(torch.LongTensor).squeeze()
            y_label_ = onehot[y_]
            y_fill_ = fill[y_]
            z_, y_label_, y_fill_ = Variable(z_.cuda()), Variable(
                y_label_.cuda()), Variable(y_fill_.cuda())

            G_result = G(z_, y_label_)
            D_result = D(G_result, y_fill_).squeeze()

            G_train_loss = BCE_loss(D_result, y_real_)

            G_optimizer.zero_grad()

            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.data)
            print('epoch [%d/%d] batch [%d/%d] - loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, i, batch_size, torch.mean(torch.FloatTensor(D_losses)),
                                                                torch.mean(torch.FloatTensor(G_losses))))

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                     torch.mean(torch.FloatTensor(G_losses))))
        fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
        show_result((epoch+1), classes, save=True, path=fixed_p)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(
        torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
    print("Training finish!... save training results")
    torch.save(G.state_dict(), root + model + 'generator_param.pkl')
    torch.save(D.state_dict(), root + model + 'discriminator_param.pkl')
    with open(root + model + 'train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path=root +
                    model + 'train_hist.png')

    images = []
    for e in range(train_epoch):
        img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)
