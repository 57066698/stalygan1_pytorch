import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm

def get_truncated_noise(n_samples, z_dim, truncation):
    """
    :param n_samples: scalar
    :param z_dim: dim of noise vector
    :param truncation: truncation value, a non-negative scalar
    :return:
    """
    truncated_noise = truncnorm.rvs(-truncation, truncation, size=(n_samples, z_dim))
    return torch.Tensor(truncated_noise)

def