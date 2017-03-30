import math
import torch
import torch.cuda.comm as comm # communication collectives, used for multi-gpu is used 
from torch.nn.parallel._functions import Broadcast # Broadcast is used to broadcast input(forward()) from input device to target gpus, and gradoutput(backward()) to the input_device 
from torch.nn.parallel import scatter, parallel_apply, gather 
import torch.nn.functional as F # provide functional interfaces: conv + conv_transposed + pool + unpool + activation + linear + bn + loss + upsample

# cast parameters from one type to another type 
def cast(params, dtype='float'):
    if isinstance(params, dict): # if params is a dictionary, then we iteratively cast 
        return {k: cast(v, dtype) for k,v in params.items()} # recursively convert it to torch.cuda.FloatTensor
    else:
        return getattr(params.cuda(), dtype)() # torch.cuda.FloatTensor
        

def conv_params(ni,no,k=1,g=1): 
    assert ni % g == 0
    return cast(torch.Tensor(no,ni/g,k,k).normal_(0,2/math.sqrt(ni*k*k)))

def linear_params(ni,no):
    return cast(dict(
        weight= torch.Tensor(no,ni).normal_(0,2/math.sqrt(ni)),
        bias= torch.zeros(no)))

def bnparams(n):
    return cast(dict(
        weight= torch.Tensor(n).uniform_(),
        bias=   torch.zeros(n)))

def bnstats(n):
    return cast(dict(
        running_mean= torch.zeros(n),
        running_var=  torch.ones(n)))

def data_parallel(f, input, params, stats, mode, device_ids, output_device=None):
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1: # only 1 device 
        return f(input, params, stats, mode)
    
    # function inside data_parallel 
    def replicate(param_dict, g):
        replicas = [{} for d in device_ids]  # replicas, list of n_devices dict
        for k,v in param_dict.iteritems():  # v is parameter
            for i,u in enumerate(g(v)):
                replicas[i][k] = u
        return replicas
    
    # broadcast parameters 
    params_replicas = replicate(params, lambda x: Broadcast(device_ids)(x))
    # broadcast stats 
    stats_replicas = replicate(stats, lambda x: comm.broadcast(x, device_ids))

    replicas = [lambda x,p=p,s=s,mode=mode: f(x,p,s,mode)
            for i,(p,s) in enumerate(zip(params_replicas, stats_replicas))]

    inputs = scatter(input, device_ids)

    outputs = parallel_apply(replicas, inputs)

    return gather(outputs, output_device)


# y: the output of the distilled model 
# teacher_scores: predictions of the teacher network, actually a distributions 
# labels: hard labels
def distillation(y, teacher_scores, labels, T, alpha):
    return F.kl_div(F.log_softmax(y/T), F.softmax(teacher_scores/T)) * (T*T * 2. * alpha) \
            + F.cross_entropy(y, labels) * (1. - alpha)

# l2 normalize on x: x / ||x||
def l2_normalize(x, dim=1, epsilon=1e-12):
    return x * x.pow(2).sum(dim).clamp(min=epsilon).rsqrt().expand_as(x)

