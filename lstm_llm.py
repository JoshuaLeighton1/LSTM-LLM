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