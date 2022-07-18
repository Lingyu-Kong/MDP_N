import torch
import numpy as np
from torch.autograd import Variable

CUDA_AVAILABLE=torch.cuda.is_available()
FLOAT=torch.cuda.FloatTensor if CUDA_AVAILABLE else torch.FloatTensor

def to_tensor(ndarray,dtype=FLOAT):
    return Variable(torch.from_numpy(ndarray)).type(dtype)
    