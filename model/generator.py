import torch
import torch.nn as nn
import torch.nn.functional as F



class G_mapping(nn.Module):
    def __init__(self,
                 latents_in,   # [N, latent_size]
                 latent_size = 512, # z_dim
                 mapping_layers = 8, # number of fc block
                 mapping_fmaps = 512, # hidden numner of fc block
                 mapping_lrmul = 0.01, # Learning
                 gain = 2**0.5, # for He init
                 use_wscale = True, # enable equalized learning rate
                 normalize_latents = True, # normalize latent vectors (Z) before feeding to mapping layers
                 dtype = torch.float32):
        super().__init__()

        latents_in = latents_in.