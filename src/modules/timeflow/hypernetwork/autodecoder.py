from typing import Any, Dict

import numpy as np

import torch
from torch import nn

class LatentToModulation(nn.Module):
    
    """Maps a latent vector to a set of modulations.
    Args:
        latent_dim (int):
        num_modulations (int):
        dim_hidden (int):
        num_layers (int):
    """

    def __init__(
        self,
        latent_dim: int,
        num_modulations: int,
        dim_hidden: int,
        num_layers: int = 1,
        activation: nn.Module = nn.SiLU()
    ) -> None:

        super().__init__()

        self.latent_dim      = latent_dim
        self.num_modulations = num_modulations
        self.dim_hidden      = dim_hidden
        self.num_layers      = num_layers
        self.activation      = activation

        if num_layers == 1:
            self.net = nn.Linear(latent_dim, num_modulations)

        else:
            layers = [nn.Linear(latent_dim, dim_hidden), self.activation]
            if num_layers > 2:
                for i in range(num_layers - 2):
                    layers += [nn.Linear(dim_hidden, dim_hidden), self.activation]
            layers += [nn.Linear(dim_hidden, num_modulations)]
            self.net = nn.Sequential(*layers)

    def forward(
        self,
        latent: torch.Tensor
    ) -> torch.Tensor:
        
        # latent: [..., latent_dim]

        out = self.net(latent) # [..., num_modulations]

        return out  # [..., num_modulations]

class LayerWiseLatentToModulation(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        dim_hidden: int,
        num_layers: int
    ) -> None:

        super().__init__()

        self.latent_dim = latent_dim
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        self.net_list = nn.ModuleList([nn.Linear(latent_dim, dim_hidden) for l in range(num_layers)])

    def forward(
        self,
        latent: torch.Tensor
    ) -> torch.Tensor:

        # latent must be (batch_size, latent_dim, num_layers) shape

        out = []

        for l in range(latent.shape[-1]):
            out.append(self.net_list[l](latent[..., l]))

        return torch.stack(out, dim=1) # [batch_size, num_layers, hidden_dim]