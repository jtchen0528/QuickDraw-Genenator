import os
import h5py
import numpy as np
from torch.utils.data import Dataset

class QuickDrawDataset_h5py(Dataset):
    def __init__(self, root, img_size, classes, samples, transform=None):

        total_data = np.empty([0, 1, img_size, img_size])
        total_label = np.empty(0)
        index = 0
        for filename in os.listdir(root)[:classes]:
            data = []
            label = []
            with h5py.File('./' + root + '/{}'.format(filename), 'r+') as f:
                data = f['rasters'][()]

            idx = np.random.randint(len(data), size=samples)
            print("Data: " + root + '/{}'.format(filename) + ", total " + str(data.shape[0]))
            data = np.reshape(data, (len(data), 1, img_size, img_size))[
                idx, :, :, :]
            for i in range(samples):
                label.append(int(index))
            label = np.reshape(label, (len(label)))

            total_data = np.append(total_data, data, axis=0)
            total_label = np.append(total_label, label, axis=0)
            index = index + 1

        self.data_frame = total_data
        self.label_frame = total_label

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        imgs = self.data_frame[idx]
        labels = self.label_frame[idx]
        return imgs, labels

def maxpool(mat, pool):
    M, N = mat.shape
    K = pool
    L = pool

    MK = M // K
    NL = N // L
    return mat[:MK*K, :NL*L].reshape(MK, K, NL, L).max(axis=(1, 3))

class QuickDrawDataset(Dataset):
    """Quick Draw dataset."""

    def __init__(self, root, label, transform=None):
        with h5py.File('./' + root + '/{}.hdf5'.format(label), 'r+') as f:
            data = f['rasters'][()]
        img_w, img_h = data.shape[1], data.shape[2]
        data = np.reshape(data, (len(data), 1, img_w, img_h))

        data2 = []

        for x in data:
            data2.append(np.array(maxpool(x[0], 2)))

        data2 = np.array(data2)
        data2 = np.reshape(data2, (len(data2), 1, img_w // 2, img_h // 2))

        self.data_frame = data2

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        imgs = self.data_frame[idx]
        return imgs