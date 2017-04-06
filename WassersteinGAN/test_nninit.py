import nninit.nninit as nninit

from torch import nn
import numpy as np

class Net(nn.Module):
  def __init__(self):
     super(Net, self).__init__()
     self.conv1 = nn.Conv2d(5, 10, (3, 3))
     nninit.xavier_uniform(self.conv1.weight, gain=np.sqrt(2))
     nninit.constant(self.conv1.bias, 0.1)

network = Net()
