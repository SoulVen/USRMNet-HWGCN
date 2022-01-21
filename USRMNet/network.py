import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from parameter_parser import parameter_parser


args = parameter_parser()


class Layer(nn.Module):
    def __init__(self, k, Fin, Fout, sigma):
        super(Layer, self).__init__()
        self.k = k
        A = t.rand(k, Fin, Fout)
        self.A = nn.parameter.Parameter(A)
        self.Fin = Fin
        self.Fout = Fout
        self.sigma = sigma

    def forward(self, H, x):
        z = self.apply_filter(H, x, self.A)
        # z = nn.BatchNorm2d(len(z), affine=True)(z)
        x = self.sigma(z)
        return x

    def apply_filter(self, H, X, A):
        k, _, _ = A.shape

        u = 0
        for i in range(k):
            HZA = X @ A[i, :, :]
            u = u + HZA
            X = t.einsum('ijk,ikl->ijl', H, X)
            # X = t.einsum('ik, kj -> ij', H, X)

        return u


class HWGCN(nn.Module):
    def __init__(self, K, F, sigma=lambda x: x):
        super(HWGCN, self).__init__()
        self.leakyrelu = nn.LeakyReLU(negative_slope=1)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        layers = []

        assert len(K) == len(F) - 1, "Incorrect number of layers"

        for layer in range(len(K)):
            if layer == len(K) - 1:
                layers.append(Layer(K[layer], F[layer], F[layer+1], self.leakyrelu))
            else:
                layers.append(Layer(K[layer], F[layer], F[layer+1], self.tanh))

        # cannot use sequential, as it only allows one parameter input
        self.layers = nn.ModuleList(layers)

    def forward(self, H, x):
        for layer in self.layers:
            x = layer(H, x)
        return x