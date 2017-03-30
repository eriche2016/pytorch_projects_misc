import torch 
from torch.autograd import Variable

# check the use of retain_variables 
x = Variable(torch.ones(2, 2), requires_grad = True)
print(x.grad)
y = x * x   
z = y * y 
print(y.requires_grad)
# y = x + 2 # inplace  
# first bp
z.backward(torch.ones(2, 2), retain_variables=True)

# the retain_variables flag will prevent the internal buffers from being freed
print "first backward of x is:"
print(x.grad)

# second bp 
y.backward(torch.ones(2, 2), retain_variables=False)
print "second backward of x is:"
print(x.grad)


