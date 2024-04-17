import os
import random
import subprocess
from os.path import join as opj

import numpy as np
import torch
import torch.nn as nn


def get_gpu_memory_usage():
    # 执行nvidia-smi命令获取GPU信息
    try:
        output = subprocess.check_output(['nvidia-smi',
                                          '--query-gpu=memory.free,memory.total',
                                          '--format=csv,nounits,noheader'])
        # 解析输出
        gpu_info = [x.split(',') for x in output.decode('utf-8').strip().split('\n')]
        gpu_memory = [(i, int(x[0]), int(x[1])) for i, x in enumerate(gpu_info)]
        min_gpu_id = max(gpu_memory, key=lambda x: x[1])[0]
        if gpu_memory:
            for (i, free, total) in gpu_memory:
                print("GPU {}: Free Memory: {:.2f} GB, Total Memory: {:.2f} GB, Utilization: {:.2f}%".\
                      format(i, free/1024, total/1024, 100 - 100.0 * free / total), end="")
                if i == min_gpu_id:
                    print("  <--- Max Free Memory GPU")
                else:
                    print()
        return gpu_memory
    except subprocess.CalledProcessError as e:
        print("Error querying nvidia-smi: ", e.output)
        return None

"""get the GPU with the minimum memory usage"""
def get_min_gpu_memory_usage():
    gpu_memory = get_gpu_memory_usage()
    if gpu_memory:
        gpu_memory.sort(key=lambda x: x[1], reverse=True)
        return gpu_memory[0][0]
    return -1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sound_normal(x):
    """
    Normalize the input sound tensor by dividing it by its standard deviation.

    Args:
        x (torch.Tensor): Input sound tensor.

    Returns:
        torch.Tensor: Normalized sound tensor.
        torch.Tensor: Standard deviation of the input sound tensor.
    """
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + 1e-8)
    return x, std

def get_params(args):
    
    params    = {}
    args_ref  = vars(args)
    args_keys = vars(args).keys()

    for key in args_keys:
        if '__' in key:
            continue
        else:
            temp_params = args_ref[key]
            if type(temp_params) == dict:
                params.update(temp_params)
            else:
                params[key] = temp_params
                
    return params
    
def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)
            
def rescale_conv(conv, reference):
    std   = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale    

def seed_init(seed=100):

    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED']       = str(seed)  


def torch_sisdr(reference, estimation):
    reference_energy = torch.sum(reference ** 2, dim=-1, keepdims=True)
    optimal_scaling = torch.sum(reference * estimation, dim=-1, keepdims=True) / reference_energy
    projection = optimal_scaling * reference
    noise = estimation - projection
    ratio = torch.sum(projection ** 2, dim=-1) / torch.sum(noise ** 2, dim=-1)
    return torch.mean(10 * torch.log10(ratio))

class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
    

