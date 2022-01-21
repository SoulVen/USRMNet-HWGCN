import torch as t
import torch.nn as nn
from utils import *


class Loss(nn.Module):
    def __init__(self, device, K, Nt, Vartheta):
        super(Loss, self).__init__()
        self.K = K                                      #  number of users
        self.Nt = Nt                                    #  antenna numbers of BS
        self.Delta = indicator(self.K).to(device)                  #  indicator matrix for each variable
        self.alpha = 1/self.K                           #  prior weight of all users, suppose all the same
        self.Vartheta = Vartheta
        self.relu = nn.ReLU
    
    def forward(self, x, ind1, ind2):
        '''
        @x:           output of the last layer, its dimmension is (batchsize, 2*K*K+3*K)
        @f_list:      最后一层迭代完成后的约束违背值
        '''
        loss = self.alpha * (t.matmul(t.log(1+x), ind1) - t.matmul(self.Vartheta*x, ind2))
        # print(loss)
        loss = t.mean(loss)

        # loss_constraint = 0
        # for i in range(len(f_list)):
        #     loss_constraint += t.sum(t.max(f_list[i], t.zeros_like(f_list[i]))) / 10
        return t.exp(-loss) # 用指数函数强行使其大于0
        # return loss_constraint, loss


def Loss_diff(K, HW, Vartheta, mu1, mu2, mu3, mu4, x1, x2, x3, x4, x5):
    '''
    This function is used to caculate the main objective function
    @K: K is the number of users
    @HW: |HW|
    @mui: Lagrangian multipiers for q, varphi, φ, ψ, θ
    @xi: indicators of q, varphi, φ, ψ, θ
    '''
    eyes = t.eye(K)
    mask = t.zeros_like(HW)
    mask[:] = eyes

    # 对q求偏导，(batchsize, K)
    tmp1 = (mu2-mu1) * HW
    tmp2 = t.sum(tmp1 * mask, 1)
    tmp3 = (((mu1-mu2)*(x2-x3)).unsqueeze(-2)) * HW
    tmp4 = t.sum(tmp3*(1-mask), 2)
    tmp5 = tmp4 * x1
    cstr1 = (tmp2 + tmp5)**2
    cstr1_mean = t.mean(t.sum(cstr1, 1), 0)  
    
    # 对varphi求偏导
    cstr2 = -1/((1+x2)*K) + mu1 * (1 + t.sum((x1.unsqueeze(-1))*HW, 1))
    cstr2_mean = t.mean(t.sum(cstr2**2, 1), 0)

    # 对theta求偏导
    cstr3 = -mu4 + Vartheta/K
    cstr3_mean = t.sum(cstr3**2)

    # 对phi求偏导
    cstr4 = 2*mu3*(1+x3)*(1-x4) - mu2*(1 + t.sum((x1.unsqueeze(-1))*HW, 1))
    cstr4_mean = t.mean(t.sum(cstr4**2, 1), 0)

    # 对psi求偏导
    cstr5 = 0.5 * mu4 * t.pow(x4, -0.5) - mu3
    cstr5_mean = t.mean(t.sum(cstr5**2, 1), 0)

    return (cstr1_mean, cstr2_mean, cstr3_mean, cstr4_mean, cstr5_mean)
    


##  test the feasibility of functions
if __name__ == "__main__":
    A = t.randn(4, 10, 44)
    b = t.ones(4, 10) * 100
    loss = Loss(4, 16, 0.337)
    x_L = t.ones(4, 44)*4
    x_L_1 = t.ones(4, 44) * 3
    mu = 3
    print(loss(x_L, x_L_1, mu, A, b))     #    test result is nan, because x doesn't meet constraints