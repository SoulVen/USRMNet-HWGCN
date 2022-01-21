import os
import scipy.io as sio
# from sympy.ntheory.residue_ntheory import _discrete_log_pohlig_hellman
import torch as t
import numpy as np
import torch.nn as nn
import torch.optim as optim
from unfold_model_pocs_seq import *
from loss_pocs import Loss
from dataset import MyDataset
from torch.utils.data import DataLoader
from utils import *
from parameter_parser import parameter_parser
from AdaptiveWeightedLoss import AdaptiveWeightedLoss
# from tqdm import tqdm
from matplotlib import pyplot as plt
import gc  # used to recycle useless memory
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu') # train model using cuda
##device = t.device('cpu')


args = parameter_parser()                           #   loading args. 
K = args.K                                          #   number of users
Nt = args.Nt                                        #   number of antennas
Vartheta = args.Vartheta                            #   the scenerio of BER=0.5 and n=128
lr = args.lr                                        #   learning rate of adam optimizer
lr_coe = args.lr_coe                                #   step-size of lagrangian multipliers
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
epoch = args.epochs
du = args.du
dc = args.dc

# loading datasets, inlcuding channel state information, initial beamforming vector, initial power
train_data_path = "USRMNet/dataset/channel{}_{}_{}_{}_train_SNR15_n256.mat".format(K, Nt, du, dc)

train_data = MyDataset(train_data_path)

USRMNet = uRLLCunfoldingNet(args, device).to(device) 
criterion = Loss(device, K, Nt, Vartheta)
optimizer_USRMNet = optim.Adam(filter(lambda p: p.requires_grad, USRMNet.parameters()), lr = lr)
awl_list = [AdaptiveWeightedLoss(2) for i in range(args.outer_itr)]

optimizer_s = optim.Adam([
                {'params': awl_list[0].parameters(), 'weight_decay': 0},
                {'params': awl_list[1].parameters(), 'weight_decay': 0},
                {'params': awl_list[2].parameters(), 'weight_decay': 0},
                {'params': awl_list[3].parameters(), 'weight_decay': 0},
                {'params': awl_list[4].parameters(), 'weight_decay': 0}	
            ], lr = lr)

loss_obj_list = []
loss_s_list = []
constraint_list = []

ind1 = t.zeros(M).to(device)
ind2 = t.zeros(M).to(device)
ind1[K:K*2] = 1
ind2[4*K:5*K] = 1

A9  =  -1*t.eye(M, M)
A1  =  -1*t.eye(M, M)
A3  =  t.eye(M, M)
A7  =  -1*t.eye(M, M)
b7  =  -(1-1/(1+nu3_val)**2)
A10 =  t.eye(M, M)


for outer_loop in range(args.outer_itr):
    for inner_loop in range(max_itr):
        # itr = itr + 1
        # set_learning_rate(optimizer, lr)
        for child in USRMNet.children():
            if outer_loop > 0:   #  If the outer layer number is not 1, the parameters of the last inner layer of the current outer layer should be frozen.
                for child_of_child in child[outer_loop-1].children():
                    for para in child_of_child[max_itr-1].parameters():
                        para.requires_grad = False
            if inner_loop > 0:   #  If the inner layer number is not 1
                for child_of_child in child[outer_loop].children():   #  Obtain all the sub-modeules of the current outer layer
                    for para in child_of_child[inner_loop-1].parameters():   #  The parameters of the previous inner layer of the current outer layer should be frozen.
                        para.requires_grad = False

        mu  =  nn.Parameter(t.zeros(4, K), requires_grad = False).to(device)   # lagrangian multipliers
        for loop in range(epoch): 
            train_dataloader = DataLoader(train_data, batchsize, shuffle=True)
            for i, data in enumerate(train_dataloader):
                data_H, w0, p0   =  data
                data_H.to(device), w0.to(device), p0.to(device)                                                                    #    prior information
                bar_gamma           =   gamma(p0, data_H, w0, sigmma)                                            #    (batchsize, K)
                tilde_gamma_val     =   tilde_gamma(Pmax, data_H.permute(0, 2, 1), sigmma).to(device)            #    (batchsize, K)
                q0                  =   p0                                                                       #    (batchsize, K)
                varphi0             =   bar_gamma                                                                #    (batchsize, K)
                phi0                =   bar_gamma                                                                #    (batchsize, K)
                psi0                =   tilde_V(phi0)                                                            #    (batchsize, K)
                theta0              =   t.sqrt(psi0)                                                             #    (batchsize, K)   
                x0                  =   t.cat([q0, varphi0, phi0, psi0, theta0], dim=1).to(device)                          #    (batchsize, 5*K)

                optimizer_USRMNet.zero_grad()
                optimizer_s.zero_grad()
                x, f_list = USRMNet(data_H, w0, x0, nu3_val, tilde_gamma_val, A9, A1, A3, A7, b7, A10, outer_loop+1, inner_loop+1)
                # print("x: ", x[0])     #    output variable 
                loss_obj = criterion(x, ind1, ind2)
                violation = t.zeros(len(f_list), K)
                violation1 = t.zeros(len(f_list), K)
                for j in range(len(f_list)):
                    violation1[j] += t.mean(f_list[j], dim=0)
                    violation[j] += t.mean(t.max(f_list[j], t.zeros_like(f_list[j])), dim=0)
                    if j == 0:
                        loss_constraint = t.sum(mu[j] * t.mean(t.max(f_list[j], t.zeros_like(f_list[j])), dim=0))
                    loss_constraint += t.sum(mu[j] * t.mean(t.max(f_list[j], t.zeros_like(f_list[j])), dim=0))
                
                loss = awl_list[outer_loop](loss_obj, loss_constraint)
                
                loss.backward()
                optimizer_USRMNet.step()
                optimizer_s.step()
                with t.no_grad():
                    loss_obj_list.append(loss_obj.item()) 
                    loss_s_list.append(loss.item())
                    violation2 = t.mean(t.mean(violation, 1))
                    constraint_list.append(violation2.item())
                    for j in range(len(f_list)):
                        mu[j] = mu[j] + lr_coe * t.max(violation1[j], t.zeros(K))

                print("Epoch {}:  loop: {}  loss_obj: {}  loss: {} constraint_loss: {}".format(outer_loop*max_itr+inner_loop, loop, loss_obj.item(), loss.item(), violation2.item()))
                gc.collect()   #   recycle memory
            
            print("mu:", mu)      #   print updated lagrangian multipliers
    t.save(USRMNet, "result/UROLL_{}_{}_SNR25_{}layer.pt".format(K, Nt, outer_loop))
    sio.savemat("result/loss_{}_{}_SNR25.mat".format(K, Nt), {"loss_obj": loss_obj_list, "loss_joint": loss_s_list, "constraint_loss": constraint_list})
