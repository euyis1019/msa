from dataclasses import dataclass, field
import random
import numpy as np
import torch

@dataclass
class Config:
    version: str = 'For Mosi 7992'
    log_dir: str = '../record'
    dropout_rate: float = 0.7
    batch_size: int = 4
    shuffle: bool = True
    learning_rate: float = 0.0001
    epochs: int = 10000
    random_seed: int = 42  # Default seed value, can be changed when creating an instance
    L1: float = 0.0000001 
    L2: float = 0.000000001
    def __post_init__(self):
        self.set_random_seed(self.random_seed)

    def set_random_seed(self, seed_value):
        print("#############")
        """Set seed for reproducibility."""
        random.seed(seed_value)        # Python random module
        np.random.seed(seed_value)     # NumPy
        torch.manual_seed(seed_value)  # CPU level seed for PyTorch
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)          # GPU level seed for PyTorch
            torch.cuda.manual_seed_all(seed_value)      # if you are using multi-GPU.
            torch.backends.cudnn.deterministic = True   # to make sure that every time you run your script on the same input, you will receive the same output (may impact performance)
            torch.backends.cudnn.benchmark = False      # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms, if this is set to False, the performance may degrade


