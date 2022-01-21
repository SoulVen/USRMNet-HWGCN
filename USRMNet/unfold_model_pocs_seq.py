import torch as t
import torch.nn as nn
import torch.nn.functional as F
from network import HWGCN
from utils import *
import numpy as np

class UnfoldPOCS(nn.Module):
    # inner Layers, do Projection Onto Convex Sets (POCS)
    def __init__(self, args, device):
        super(UnfoldPOCS, self).__init__()
        self.device       =    device
        self.K            =    args.K                                                           #   number of users
        self.alpha_k      =    1/self.K                                                         #   prior weight of users
        self.Nt           =    args.Nt                                                          #   number of antennas
        self.batchsize    =    args.batchsize                                                   #   batchsize
        self.N            =    10*self.K+1                                                      #   number of affine constraints
        self.M            =    5*self.K                                                         #   number of variables
        self.delta        =    indicator(self.K).to(self.device)                                                #   indicator vector
        self.vartheta     =    args.Vartheta                                                    #   vartheta value
        self.D            =    args.D                                                           #   bytes of transimitted data
        self.n            =    args.n                                                           #   finite blocklength
        self.sigmma       =    args.sigmma                                                      #   gaussian noise
        self.lbd          =    nn.Parameter(1.0*t.ones(2), requires_grad=True).to(self.device)     #   define learnable lambda variable
        self.mu           =    nn.Parameter(1.0*t.ones(11), requires_grad=True).to(self.device)     #   define learnable lambda variable
        self.indicatorq   =    self.indicator_q().to(self.device)                                               #   indicator of variable q
        self.Pmax         =    args.Pmax                                                          #   limits of maximal power
        self.adj          =    t.ones(args.batchsize, self.K, self.K)                               # 邻接矩阵
        self.relu         =    nn.ReLU()
        self.F1           =    [5, 32, 2]
        self.F2           =    [5+self.K, 32, 5]
        self.K0           =    [1, 1]
        self.REGNNs       =    nn.ModuleList([HWGCN(self.K0, self.F1, self.relu), HWGCN(self.K0, self.F2, self.relu)]).to(self.device)
    
    def proj_C8(self, x):
        # projection operation for constarint (8)
        '''
        @x:             当前变量值, (batchsize, self.M)
        '''
        Tmp = x@self.indicatorq         # (batchsize, )
        PC_ = t.zeros_like(x).to(self.device)
        f = Tmp - self.Pmax       # (batchsize, )

        for j in range(self.batchsize):
            tmp_left = f[j]
            tmp_x = x[j]
            if tmp_left <= 0:
                PC_[j] = tmp_x
            else:
                x_tp = tmp_x + (-tmp_left)*self.indicatorq / t.norm(self.indicatorq)**2
                #x_ = tmp_x - mu * (tmp_x - x_tp)
                PC_[j] = x_tp

        return f, PC_
    
    
    def indicator_q(self):
        indicator_q = t.zeros(self.M).to(self.device)
        indicator_q[0:self.K] = 1                        #  (5*K)
        return indicator_q


    def cstr_5(self, x1, x2):
        f5 = 1-1/(1+x2)**2 - x1
        return f5

    def cstr_6(self, x1, x2):
        f6 = t.sqrt(x2) - x1
        return f6

    def cstr_4(self, HW, x1, x2, x3):
        # self.cstr_4(HW, x[:, 2*self.K:3*self.K], x[:, 0:self.K], x[:, self.K:2*self.K])
        eyes = t.eye(self.K)
        mask = t.zeros_like(HW)
        mask[:] = eyes
        tmp = x2.unsqueeze(-1)*HW[:,:,:]
        tmp_top = t.sum(tmp * mask, 2)
        tmp_down = t.sum(tmp * (1-mask), 1) + 1
        # gamma_k = tmp_top / tmp_down

        f4 = tmp_top - tmp_down*x1
        f2 = x3*tmp_down - tmp_top
        return f4, f2
        
    
    # def forward(self, HW, x0_tmp, A6, b6, nu3_val, tilde_gamma_val, max_itr):
    def forward(self, num_itr, kth, HW, x0_tmp, nu3_val, tilde_gamma_val, A9, A1, A3, A7, b7, A10):
        '''
        @HW:                    abs(H*W),  (batchsize, K, K)
        @x0_tmp:                initial varible vector,  (batchsize, K)
        @nu3_val:               constant
        @tilde_gamma_val:       constant
        @max_itr:               layer of current iteration
        '''
        f_list = []   #  存储最后一次迭代得到的f， 即约束条件的违背值

        x = x0_tmp.to(self.device)

        lbd = self.REGNNs[0](HW, (t.reshape(x, (self.batchsize, 5, self.K))).permute(0, 2, 1)).to(self.device)
        lbd = self.relu(t.mean(lbd, dim=1))

        # projection gradient descent
        for i in range(self.K):
            x_K_tmp = x[:, self.K+i]
            x_4K_tmp = x[:, 4*self.K+i]
            x[:, self.K+i] = x_K_tmp - lbd[:, 0] * (self.alpha_k / (1+x_K_tmp))
            x[:, 4*self.K+i] = x_4K_tmp - lbd[:, 1] * (-self.alpha_k*self.vartheta)
        
        tilde_x = self.REGNNs[1](HW, t.cat((HW, (t.reshape(x, (self.batchsize, 5, self.K))).permute(0, 2, 1)), dim=-1)).to(self.device)
        tilde_x = t.reshape((tilde_x.permute(0, 2, 1)), (self.batchsize, self.M))
        x = x + tilde_x

        # constarint (8)
        _, x_tmp = self.proj_C8(x)
        x = x_tmp

        #  constraint (1)
        tmp_nu = t.zeros_like(x)
        tmp_nu[:, self.K:2*self.K] = nu3_val
        x = t.max(tmp_nu, x)

        #  constraint (3)
        tmp_gamma = t.ones_like(x) * 1e9
        tmp_gamma[:, 2*self.K:3*self.K] = tilde_gamma_val
        x = t.min(tmp_gamma, x)

        #  constraint (7-left)
        tmp_left = t.zeros_like(x)
        tmp_left[:, 3*self.K:4*self.K] = (1-1/(1+nu3_val)**2)
        x = t.max(tmp_left, x)
        
        #  constarint (7-right)
        V_gamma = 1-1/(1+tilde_gamma_val)**2
        tmp_right = t.ones_like(x) * 1e9
        tmp_right[:, 3*self.K:4*self.K] = V_gamma
        x = t.min(tmp_right, x)

        if kth == num_itr-1:
            x = t.max(x, t.zeros_like(x))
            
            #  constraint (5)
            f5 = self.cstr_5(x[:, 3*self.K:4*self.K], x[:, 2*self.K:3*self.K])
            f_list.append(f5)
            
            #  constraint (6)
            f6 = self.cstr_6(x[:, 4*self.K:5*self.K], x[:, 3*self.K:4*self.K])
            f_list.append(f6)

            #  constraint (4)
            f4, f2 = self.cstr_4(HW, x[:, 2*self.K:3*self.K], x[:, 0:self.K], x[:, self.K:2*self.K])
            f_list.append(f4)
            f_list.append(f2)
        return x, f_list


class PowerControlNet(nn.Module):
    def __init__(self, args, device):
        super(PowerControlNet, self).__init__()
        self.K           =   args.K
        self.device      =   device                                                  #  number of users
        self.Nt          =   args.Nt                                                 #  number of antennas
        self.batchsize   =   args.batchsize                                          #  batchsize
        self.N           =   10*self.K+1                                             #  number of affine constraints
        self.M           =   5*self.K                                                #  number of variables
        self.max_itr     =   args.max_itr
        self.Layers      =   nn.ModuleList([UnfoldPOCS(args, self.device) for i in range(self.max_itr)]).to(self.device)

    def forward(self, HW, x0_tmp, nu3_val, tilde_gamma_val, A9, A1, A3, A7, b7, A10, num_itr):
        x = x0_tmp
        f_list_out = []
        for i in range(num_itr):
            x_tmp, f_list = self.Layers[i](num_itr, i, HW, x, nu3_val, tilde_gamma_val, A9, A1, A3, A7, b7, A10)
            x = x_tmp
            if i == num_itr-1: 
                f_list_out = f_list

        return x, f_list_out


class uRLLCunfoldingNet(nn.Module):
    def __init__(self, args, device):
        super(uRLLCunfoldingNet, self).__init__()
        self.args         =   args
        self.device       =   device
        self.K            =   args.K                                                  #  number of users
        self.Nt           =   args.Nt                                                 #  number of antennas
        self.batchsize    =   args.batchsize                                          #  batchsize
        self.N            =   10*self.K+1                                             #  number of affine constraints
        self.M            =   5*self.K                                                #  number of variables
        self.outer_itr    =   args.outer_itr
        self.max_itr      =   args.max_itr
        self.outerLayers  =   nn.ModuleList([PowerControlNet(args, self.device) for _ in range(self.outer_itr)]).to(self.device)

    def forward(self, data_H, w0, x0_tmp, nu3_val, tilde_gamma_val, A9, A1, A3, A7, b7, A10, outer_itr, inner_itr):

        x = x0_tmp
        HW =  t.pow(t.abs(t.bmm(data_H, w0)), 2).to(self.device)
        
        f_list_out = []

        if outer_itr > 1:
            with t.no_grad():
                for j in range(outer_itr-1):
                    x_tmp, f_list = self.outerLayers[j](HW, x, nu3_val, tilde_gamma_val, A9, A1, A3, A7, b7, A10, self.max_itr)
                    # x = x_tmp.detach()
                    x = x_tmp
                    w0 = self.update_W(x, data_H).to(self.device)
                    HW =  t.pow(t.abs(t.bmm(data_H, w0)), 2).to(self.device) #    (batchsize, K, K)
        
        x_tmp, f_list = self.outerLayers[outer_itr-1](HW, x, nu3_val, tilde_gamma_val, A9, A1, A3, A7, b7, A10, inner_itr)
        x = x_tmp
        with t.no_grad():
           w0 = self.update_W(x, data_H)
        f_list_out = f_list

        return x, f_list_out


    def update_W(self, x_k_half, data_H):

        x_k_half = x_k_half.to(t.device('cpu'))
        data_H = data_H.to(t.device('cpu')) 
        W_star_tau = t.zeros((self.args.batchsize, self.args.Nt, self.args.K), dtype=t.complex64) 
        obj_outer = []
        for loop in range(self.args.batchsize):
            #    compute W_star
            W_star_tau_tmp = t.zeros((self.args.Nt, self.args.K), dtype=t.complex64)
            q = x_k_half[loop, 0:self.args.K]      #    obtain the uplink power 

            hh = t.zeros(1)
            for i in range(self.args.K):
                hh = hh + q[i]*data_H[loop,i].unsqueeze(-1)@data_H[loop,i].conj().unsqueeze(0)
            
            hh = hh + t.eye(self.args.Nt)
            hh = t.inverse(hh)

            for i in range(self.args.K):
                tmp = (hh @ (data_H[loop,i].T)) / (np.linalg.norm(hh@(data_H[loop,i].T).numpy(), 2))
                W_star_tau_tmp[:, i] = tmp
            W_star_tau_tmp = W_star_tau_tmp.conj()
            W_star_tau[loop]  =  W_star_tau_tmp

            keshi = 0    # initialize objective value for equation (7a)
            for i in range(self.args.K):
                tmp_top = q[i] * np.abs(data_H[loop, i]@W_star_tau_tmp[:, i])**2
                tmp_down = 0
                for j in range(self.args.K):
                    if i != j:
                        tmp_down = tmp_down + q[j]*np.abs(data_H[loop, j]@W_star_tau_tmp[:, i])**2
                gamma_tmp = tmp_top / (tmp_down + 1)
                keshi = keshi + (np.log(gamma_tmp+1) - self.args.Vartheta*np.sqrt(1-1/((1+gamma_tmp)**2)))/self.args.K    #  caculate the outer objective value
            
            obj_outer.append(keshi.item())
        return  W_star_tau