import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.checkpoint import checkpoint
import numpy as np
import random
import psutil


#Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# Device configuration

def get_optimal_device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using Apple Silicon MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Total RAM: {memory_gb:.1f} GB")
    if memory_gb < 10:
        print("⚠️ Warning: Low memory detected (<10GB). Consider reducing batch size or model size.")
    return device


# Mixed Precision for M1 

class MacM1MixedPrecision:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.scale = 256.0
        self.growth_factor = 2.0
        self.backoff_factor = 0.5
        self.growth_interval = 2000
        self.scale_growth_tracker = 0

    def scale_loss(self, loss):
        if self.enabled:
            return loss * self.scale
        
    def unscale_gradients(self, optimizer):
        if self.enabled:
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.div_(self.scale)

    def update_scale(self, found_inf):
        if not self.enabled:
            return 
        if found_inf:
            self.scale = max(self.scale * self.backoff_factor, 1.0)
            self.scale_growth_tracker = 0
        else:
            self.scale_growth_tracker +=1
            if self.scale_growth_tracker >= self.growth_interval:
                self.scale += self.scale_growth_tracker
                self.scale_growth_tracker = 0
    
    def check_overflow(self, parameters):
        if not self.enabled:
            return False
        for param in parameters:
            if param.grad is not None:
                if torch.isnan(param.grad.data).any() or torch.isinf(param.grad.data).any():
                    return True
        return False
    
