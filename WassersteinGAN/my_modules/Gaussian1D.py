# ref: https://discuss.pytorch.org/t/how-to-define-a-new-layer-with-autograd/351
# given x 
# this new layer compute its gaussian probability 
# density value: f(x) = a * exp^((x-b)^2 / c)
# where a, b, c are the parameters and need to be updated 
import torch
import torch.nn as nn 
from torch.autograd import Variable 

class Gaussian(nn.Module): 
    def __init__(self):
        super(Gaussian, self).__init__() 

        self.a = nn.Parameter(torch.zeros(1)) 
        self.b = nn.Parameter(torch.zeros(1)) 
        self.c = nn.Parameter(torch.ones(1)) 

    def forward(self, x):
        # unfortunately we donot have automatic broadcasting yet 
        a = self.a.expand_as(x) 
        b = self.b.expand_as(x) 
        c = self.c.expand_as(x) 
        return a * torch.exp((x-b) * (x-b) / c) 


