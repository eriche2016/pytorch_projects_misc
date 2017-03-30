import torch
from torch.autograd import Variable
from Gaussian1D import Gaussian


module = Gaussian() 
x = Variable(torch.randn(20)) 
out = module(x) 
print(out) 

'''
loss = loss_fn(out) 
loss.backward() 

# now module.a.grad should be non-zero 

# after inserting Gaussian Module into net
optimizer = optim.SGD(net.parameters(), lr=0.01) 
optimizer.zero_grad() 
output = net(input) 
loss = criterion(output, target) 
loss.backward() 
optimizer.step() 
'''
