import os
# from termios import FF1
import scipy.io as sio
# from sympy.ntheory.residue_ntheory import _discrete_log_pohlig_hellman
import torch as t
import numpy as np
import torch.nn as nn
import torch.optim as optim
from unfold_model_pocs_seq_test import *
from loss_test import Loss
from dataset import MyDataset
from torch.utils.data import DataLoader
from utils import *
from parameter_parser import parameter_parser
# from tqdm import tqdm
from matplotlib import pyplot as plt
import gc  # used to recycle useless memory
import time

args = parameter_parser()                           #   loading args. 
K = args.K                                          #   number of users
Nt = args.Nt                                        #   number of antennas
Vartheta = args.Vartheta                            #   the scenerio of BER=0.5 and n=128
lr = args.lr                                        #   learning rate of adam optimizer
lr_coe = args.lr_coe                                #   step-size of lagrangian multipliers
epochs = args.epochs                                #   training epoches
batchsize = args.batchsize                          #   training batchsize
sigmma = args.sigmma                                #   gaussian noise
D = args.D                                          #   the size of data transimitted
n = args.n                                          #   finite blocklength
nu3_val = nu3(D, n, Vartheta)                       #   the value of nu_3
Pmax = args.Pmax                                    #   limits of maximal power
N = 10*K+1                                          #   number of constraints
M = 5*K                                             #   number of variables
num_H = args.num_H                                  #   number of training samples
max_itr = args.max_itr                              #   maximum iterations of outer layer

# loading datasets, inlcuding channel state information, initial beamforming vector, initial power
test_data_path = "dataset/channel{}_{}_140_180_test_SNR15_n{}.mat".format(K, Nt, n)

test_data = MyDataset(test_data_path)
# test_data = sio.loadmat(test_data_path)

model = t.load("result\\UROLL_{}_{}_SNR15_n{}_100_140\\UROLL_{}_{}_SNR15_n{}_100_140_1layer_20000.pt".format(K, Nt, n, K, Nt, n))
criterion = Loss(K, Nt, Vartheta)   # original objective fucntion, i.e., equation (10a)

ind1 = t.zeros(M)
ind2 = t.zeros(M)
ind1[K:K*2] = 1
ind2[4*K:5*K] = 1

A9  =  -1*t.eye(M, M)
A1  =  -1*t.eye(M, M)
A3  =  t.eye(M, M)
A7  =  -1*t.eye(M, M)
b7  =  -(1-1/(1+nu3_val)**2)
A10 =  t.eye(M, M)

constraint_loss = []

test_dataloader = DataLoader(test_data, batchsize, shuffle=False)

print("start")
for outer_loop in  [1]: # range(args.outer_itr):  
    for inner_loop in [1]: # range(max_itr):
        start = time.time()
        for i, data in enumerate(test_dataloader):
            data_H, w0, p0   =  data                                                                     #    prior information
            bar_gamma        =  gamma(p0, data_H, w0, sigmma)                                            #    (batchsize, K)
            tilde_gamma_val  =  tilde_gamma(Pmax, data_H.permute(0, 2, 1), sigmma)                       #    (batchsize, K)
            q0               =  p0                                                                       #    (batchsize, K)
            varphi0          =  bar_gamma                                                                #    (batchsize, K)
            phi0             =  bar_gamma                                                                #    (batchsize, K)
            psi0             =  tilde_V(phi0)                                                            #    (batchsize, K)
            theta0           =  t.sqrt(psi0)                                                             #    (batchsize, K)   
            x0               =  t.cat([q0, varphi0, phi0, psi0, theta0], dim=1)                          #    (batchsize, 5*K)

            x, w1, f_list, obj_USRMax = model(data_H, w0, x0, nu3_val, tilde_gamma_val, A9, A1, A3, A7, b7, A10, 1, 1)
            violation = np.zeros(batchsize)
            for j in range(len(f_list)):
                violation += t.sum(t.max(f_list[j], t.zeros_like(f_list[j])), dim=1).detach().numpy()
            
            constraint_loss.extend(violation/(4*K))
            loss_obj = criterion(x, ind1, ind2)
            print("Epoch {}:".format(outer_loop*max_itr+inner_loop))
            
            print("Epoch {}:   loss_obj: {} ".format(outer_loop*max_itr+inner_loop, loss_obj.item()))

sio.savemat("result\\UROLL_{}_{}_SNR15_n{}_100_140\\obj_USRMNet_{}_{}_140_180_SNR15_n{}_1layer_20220121.mat".format(K, Nt, n, K, Nt, n), {"obj_USRMax": obj_USRMax, "constraint_loss": constraint_loss})
