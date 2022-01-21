import os
import numpy as np
import scipy.io as sio 
from torch.utils.data import Dataset, DataLoader
import h5py
import torch


class MyDataset(Dataset):
    def __init__(self, filename):
        super(MyDataset).__init__()
        data = sio.loadmat(filename)
        self.data_H = data["H"]
        self.w0 = data["w0"]
        self.p0 = data["p0"]

    def __len__(self):
        return len(self.data_H)

    def __getitem__(self, idx):
        data_H = self.data_H[idx]
        w0 = self.w0[idx]
        p0 = self.p0[idx]
        return (data_H.astype(np.complex64), w0.astype(np.complex64), p0.astype(np.float32))



if __name__ == "__main__":
    filename = "dataset/feature_train_20.mat"
    traindatset = MyDataset(filename)
    traindataloader = DataLoader(traindatset, 50)
    for i, data in enumerate(traindataloader):
        print(data[2].shape)
