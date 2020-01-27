"""=============================================================================
CUDA-related utility functions.
============================================================================="""

import torch

DEBUG=0

def device():
    """Return current CUDA device if on GPUs else CPU device.
    """
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    else:
        return torch.device('cpu')
