import os
import numpy as np
import random
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    """
    Fix all random seeds we use for reproducibility.
    """
    # Python random seed
    random.seed(seed)
    # Numpy random seed
    np.random.seed(0)
    # PyTorch random seed
    torch.manual_seed(seed)

    if device == "cuda":
        # If you are using CUDA
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

        # PyTorch backend settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
