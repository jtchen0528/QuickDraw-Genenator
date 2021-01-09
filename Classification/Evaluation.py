import torch
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import matplotlib.pyplot as plt
import torch.utils.data as data
import pandas as pd
import math
import argparse

from tqdm import tqdm
from random import randint

from DataUtils.load_data import QD_Dataset

from models import resnet34
from models import convnet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick, Draw! data Evaluation for final project.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_classes', '-class', type=int, default=30,
                        help='# of classes to classify.')
    parser.add_argument('--data_root', '-root', type=str, default='Evaluation',
                        help='root for the evaluated result directory.')
    parser.add_argument('--input', '-i', type=str, default='test.npy',
                        help='input npy for classification.')
    parser.add_argument('--ngpu', type=int,
                        default=1, help='0 or less for CPU.')
    parser.add_argument('--model', '-m', type=str,
                        default='resnet34', help='choose the model.')
    parser.add_argument('--eval_bs', '-eb', type=int,
                        default=64, help='evaluation batch size.')
    parser.add_argument('--model_file_name', '-name', type=str,
                        default='Trained_Models/model.pytorch', help='name of saved model')

    args = parser.parse_args()

    os.makedirs(args.data_root, exist_ok=True)

    def generate_eval_dataset(rawdata_root=args.data_root, target_root="Dataset"):
        """
        args:
        - rawdata_root: str, specify the directory path of raw data
        - target_root: str, specify the directory path of generated dataset
        - vfold_ratio: float(0-1), specify the test data / total data
        - max_item_per_class: int, specify the max items for each class
            (because the number of items of each class is far more than default value 5000)
        - show_imgs: bool, whether to show some random images after generation done
        """

        # Create the directories for download data and generated dataset.
        if not os.path.isdir(os.path.join("./", rawdata_root)):
            os.makedirs(os.path.join("./", rawdata_root))
        if not os.path.isdir(os.path.join("./", target_root)):
            os.makedirs(os.path.join("./", target_root))

        print("*"*50)
        print("Generate dataset from npy data")
        print("*"*50)
        all_files = glob.glob(os.path.join(rawdata_root, '*.npy'))
        print("Classes number: "+str(len(all_files)))
        print("*"*50)

        # initialize variables
        x = np.empty([0, 784])
        y = np.empty([0])
        class_names = []
        class_samples_num = []

        # load each data file
        for idx, file in enumerate(all_files):
            data = np.load(file)
            # print(data.shape)

            indices = np.arange(0, data.shape[0])
            # randomly choose max_items_per_class data from each class
            data = data[indices]

            # print(data.shape)
            labels = np.full(data.shape[0], -1)

            x = np.concatenate((x, data), axis=0)
            y = np.append(y, labels)

            class_name, ext = os.path.splitext(os.path.basename(file))
            class_names.append(class_name)
            class_samples_num.append(str(data.shape[0]))

            print(str(idx+1)+"/"+str(len(all_files)) +
                  "\t- "+class_name+" has been loaded. \n\t\t Totally "+str(data.shape[0])+" samples.")
            print("~"*50)

        print("\n"+"*"*50)
        print("Data loading done.")
        print("*"*50)
        data = None
        labels = None

        print("x size: ")
        print(x.shape)

        np.savez_compressed(target_root+"/eval", data=x, target=y)

        print("*"*50)
        print("Great, data_cache has been saved into disk.")
        print("*"*50)

        return True

    def eval():
        result = []
        net.eval()
        # info printed in terminal
        data_loader = tqdm(eval_loader, desc='Testing')
        # data_loader = test_loader  # info log in logtext
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = torch.autograd.Variable(data.cuda()), \
                torch.autograd.Variable(target.cuda())        # else:
            #     data = torch.autograd.Variable(data.cpu()), \
            #         torch.autograd.Variable(target.cpu())

            data = data.view(-1, 1, 28, 28)
            data /= 255.0

            # forward
            output = net(data)

            # accuracy
            pred = output.data.max(1)[1]
            result.append(pred)
        return result

    def saveImg(root, canvas, name, idx):
        shape = int(math.sqrt(canvas.shape[0]))
        Img = np.zeros((shape, shape, 3))
        for i in range(shape):
            for j in range(shape):
                if canvas[i * shape + j] == 0:
                    Img[i][j] = [1, 1, 1]
                else:
                    Img[i][j] = [1 - canvas[i * shape + j]/255, 1 -
                                 canvas[i * shape + j]/255, 1 - canvas[i * shape + j]/255]
        plt.imshow(Img)
        plt.title(name)
        plt.savefig("./" + root + "/ans/" + str(idx) + "_" + name + ".png")

    net = None
    if args.model == 'resnet34':
        net = resnet34(args.num_classes)
    elif args.model == 'convnet':
        net = convnet(args.num_classes)

    if args.ngpu > 1:
        net = nn.DataParallel(net)

    if args.ngpu > 0:
        net.cuda()

    net.load_state_dict(torch.load('./' + args.model_file_name))
    print("Model loaded, start evaluating.")

    generate_eval_dataset()

    eval_data = QD_Dataset(mtype="eval", root="Dataset")
    eval_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=args.eval_bs, shuffle=False)

    res = eval()
    print("Evaluation finished, start generating csv.")

    total = 0
    evaluation_res = np.empty(0)
    for r in res:
        detached = r.detach().cpu().numpy()
        evaluation_res = np.append(evaluation_res, detached)
        total = len(detached) + total
    evaluation_res = [int(i) for i in evaluation_res.tolist()]

    labels = []
    with open("./DataUtils/class_names.txt", "r") as f:
        for line in f:
            line = line[12:]
            line = line.split('\t')[0]
            labels.append(line)

    ans = []

    for item in evaluation_res:
        ans.append(labels[item])

    obj = {"Answer": ans}
    cols = ["Answer"]
    df = pd.DataFrame(obj, columns=cols)

    df.to_csv('./' + args.data_root + '/ans.csv')
    print("Csv generation complete. start generating graphs.")

    if not os.path.isdir(os.path.join("./" + args.data_root + "/", "ans")):
        os.makedirs(os.path.join("./" + args.data_root + "/", "ans"))

    Imgpath = "./" + args.data_root + "/" + "ans"
    files = os.listdir(Imgpath)
    for f in files:
        os.remove(Imgpath + "/" + f)

    TestData = np.load('./' + args.input)
    index = 0
    for i in range(len(TestData)):
        index = index + 1
        if (index % 20 == 0):
            print("Graph generation at " + str(int(index /
                                                   len(TestData) * 10000) / 100) + " Percent.", end='\r')
        saveImg(args.data_root, TestData[i], df['Answer'][i], i)

    print("Evaluation completed.")
