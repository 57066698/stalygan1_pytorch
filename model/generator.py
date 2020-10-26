import torch
import torch.nn as nn
import torch.nn.functional as F




class DenseBlock(nn.Module):
    def __init__(self, in_dim, out_dim, gain=2**(0.5), lrmul=1.0, use_swcale=True, bias=True):
        super(DenseBlock, self).__init__()
        he_std = gain * in_dim ** (-0.5) ##  HE init:  v**2 = 2/N   =>  v = (2**0.5) * (N ** 0.5)



class G_mapping(nn.Module):
    def __init__(self,
                 latents_in,   # [N, latent_size]
                 latent_size = 512, # z_dim
                 mapping_layers = 8, # number of fc block
                 mapping_fmaps = 512, # hidden numner of fc block
                 mapping_lrmul = 0.01, # Learning
                 normalize_latents = True, # normalize latent vectors (Z) before feeding to mapping layers
                 dtype = torch.float32):
        super().__init__()
