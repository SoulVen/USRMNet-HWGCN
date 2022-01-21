import torch as t
import torch.nn as nn

class AdaptiveWeightedLoss(nn.Module):
    def __init__(self, num):
        super(AdaptiveWeightedLoss, self).__init__()
        params = t.ones(num, requires_grad=True)
        self.params = t.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum +=  t.exp(-0.5*self.params[i]) * loss + t.exp(0.5*self.params[i])
        return loss_sum

if __name__ == '__main__':
    awl = AdaptiveWeightedLoss(2)
    print(awl.parameters())