import torch as t
import torch.nn as nn
from utils import *


class Loss(nn.Module):
    def __init__(self, K, Nt, Vartheta):
        super(Loss, self).__init__()
        self.K = K                                      #  number of users
        self.Nt = Nt                                    #  antenna numbers of BS
        self.Delta = indicator(self.K)                  #  indicator matrix for each variable
        self.alpha = 1/self.K                           #  prior weight of all users, suppose all the same
        self.Vartheta = Vartheta
        self.batchsize = 10
    
    def forward(self, x, ind1, ind2):
        '''
        @x:           output of the last layer, its dimmension is (batchsize, 2*K*K+3*K)
        
        '''
        loss = self.alpha * t.mean(t.matmul(t.log(1+x), ind1) - t.matmul(self.Vartheta*x, ind2))
        return -loss


## test the feasibility of functions
if __name__ == "__main__":
    A = t.randn(4, 10, 44)
    b = t.ones(4, 10) * 100
    loss = Loss(4, 16, 0.337)
    x_L = t.ones(4, 44)*4
    x_L_1 = t.ones(4, 44) * 3
    mu = 3
    print(loss(x_L, x_L_1, mu, A, b))     #    test result is nan, because x doesn't meet constraints