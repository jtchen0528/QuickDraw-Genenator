import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick, Draw! data strokes to graph transformation script',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_root', '-root', type=str, default='data',
                            help='root for the csv directory.')

    parser.add_argument('--data_target', '-target', type=str, default='data_npy',
                            help='root for the target npy directory.')

    parser.add_argument('--data_samples', '-samples', type=int, default=5000,
                            help='# of samples to translate.')

    parser.add_argument('--data_random', '-random', type=int, default=1,
                            help='randomize csv data.')

    parser.add_argument('--data_shrink_rate', '-sr', type=int, default=9,
                            help='graph shrink rate.')

    def generateGraphs(data, size, name, sampleNum, random):
        if (random == 1):
            data = np.array(data['drawing'].sample(sampleNum))
        else: 
            data = np.array(data['drawing'][:sampleNum])
        firstGraph = stroke2graph(data[0], size)
        graphList = firstGraph.reshape(1, firstGraph.shape[0] * firstGraph.shape[1])
        i = 1
        for d in data[1:]:
            canvas = stroke2graph(d, size)
            canvas = canvas.reshape(1, canvas.shape[0] * canvas.shape[1])
            graphList = np.append(graphList, canvas, axis = 0)
            i = i + 1
            if (i % 100 == 0):
                print(name + " " + str(int(i / len(data) * 10000) / 100) + " percent completed", end='\r')
        print('\n')
        print(name + " transform completed")
        return graphList

    def stroke2graph(strokes, size):
        strokes = eval(strokes)
        return makeGraph(255, strokes, size).astype(int)

    def makeGraph(size, strokes, graphsize):
        canvas = np.zeros((size + 1, size + 1))
        for stroke in strokes:
            canvas = addStroke(canvas, stroke, 5)
        canvas = maxPool(canvas, graphsize)
        canvas = np.rot90(canvas, -1)
        # showImg(canvas)
        return canvas

    def maxPool(mat, size):
        M, N = mat.shape
        K = size
        L = size

        MK = M // K
        NL = N // L
        return (mat[:MK*K, :NL*L].reshape(MK, K, NL, L).max(axis=(1, 3)))

    def addStroke(canvas, stroke, width):
        for i in range(len(stroke[0]) - 1):
            pointX1 = stroke[0][i]
            pointY1 = stroke[1][i]
            pointX2 = stroke[0][i + 1]
            pointY2 = stroke[1][i + 1]
            if (pointY2 - pointY1) == 0:
                for x in range(pointX2, pointX1):
                    canvas = drawWidth(canvas, x, pointY1, width)
            elif (pointX2 - pointX1) == 0:
                for y in range(pointY2, pointY1):
                    canvas = drawWidth(canvas, pointX1, y, width)
            else:
                if abs(pointY2 - pointY1) < abs(pointX2 - pointX1):
                    m = int((pointX2 - pointX1) / (pointY2 - pointY1))
                    if pointY2 - pointY1 > 0: step = -1
                    else: step = 1
                    x = pointX2
                    for y in range(pointY2, pointY1, step):
                        for i in range(1, abs(m) + 1, 1):
                            if pointX1 > pointX2:
                                if x + 1 < canvas.shape[0]:
                                    canvas = drawWidth(canvas, x + 1, y, width)
                                    x = x + 1
                            else:
                                if x - 1 >= 0:
                                    canvas = drawWidth(canvas, x - 1, y, width)
                                    x = x - 1
                else:
                    m = int((pointY2 - pointY1) / (pointX2 - pointX1))
                    if pointX2 - pointX1 > 0: step = -1
                    else: step = 1
                    y = pointY2
                    for x in range(pointX2, pointX1, step):
                        for i in range(1, abs(m) + 1, 1):
                            if pointY1 > pointY2:
                                if y + 1 < canvas.shape[1]:
                                    canvas = drawWidth(canvas, x, y + 1, width)
                                    y = y + 1
                            else:
                                if y - 1 >= 0:
                                    canvas = drawWidth(canvas, x, y - 1, width)
                                    y = y - 1
        return canvas

    def showImg(canvas):
        Img = np.zeros((canvas.shape[0], canvas.shape[1], 3))
        for i in range(canvas.shape[0]):
            for j in range(canvas.shape[1]):
                if canvas[i][j] == 0:
                    Img[i][j] = [1, 1, 1]
                else: 
                    Img[i][j] = [1 - canvas[i][j]/255, 1 - canvas[i][j]/255, 1 - canvas[i][j]/255]
        plt.imshow(Img)

    def drawWidth(canvas, x, y, width):
        w = int((width - 1) / 2)
        canvas[x][y] = 255
        for i in range (w, 1, -1):
            for x_s in range(x - i, x + i, 1):
                for y_s in range(y - i, y + i, 1):
                    if x_s < canvas.shape[0] and x_s >= 0 and y_s < canvas.shape[1] and y_s >= 0:
                        if canvas[x_s][y_s] < 200 / i + 55: canvas[x_s][y_s] = 200 / i + 55
        return canvas

    args = parser.parse_args()

    if not os.path.isdir(os.path.join("./" + args.data_root)):
        os.makedirs(os.path.join("./" + args.data_root))
    if not os.path.isdir(os.path.join("./" + args.data_target)):
        os.makedirs(os.path.join("./" + args.data_target))

    currentPath = os.getcwd()
    path = currentPath + "/" + args.data_root
    files = os.listdir(path)
    print(files)
    for f in files:
        fullpath = os.path.join(path, f)
        df = pd.read_csv(fullpath)
        if len(df['drawing']) < args.data_samples: args.data_samples = len(df['drawing'])
        np.save('./' + args.data_target + '/' + f[:-4], generateGraphs(df, args.data_shrink_rate, f[:-4], args.data_samples, args.data_random))