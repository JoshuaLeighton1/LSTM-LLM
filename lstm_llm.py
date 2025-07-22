import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
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

device = get_optimal_device()


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
    

class EnhancedNextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=512, num_layers=3, dropout=0.3):
        super(EnhancedNextWordLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, param in self.lstm.named_parameters():
            if 'weights_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param,0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def lstm_segment(self, embedded, h0, c0):
        return self.lstm(embedded, (h0, c0))
    
    def forward(self, x, use_mixed_precision=True, use_checkpoint=True):
        if use_mixed_precision and device.type == 'cuda':
            with autocast():
                return self._forward_impl(x, use_checkpoint)
        else:
            return self._forward_impl(x, use_checkpoint)