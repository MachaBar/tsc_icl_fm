from typing import Tuple, List

import torch
import torch.nn as nn

from einops import rearrange

class LocalityAwareINRDecoder(nn.Module):
    """ INR decoder taking already mutli-scale pos-embedded coords as inputs. """

    def __init__(
        self,
        output_dim: int = 1,
        embed_dim: int = 16,
        scales: List[int] = [3, 4, 5],
        dim: int = 128,
        depth: int = 3,
        start_quantile: float = 0.05,
        end_quantile: float = 0.95,
        nb_quantiles: int = 19,
        quantile_median: int = 9
    ) -> None:
        """
        Args:
            output_dim (int): output dim of the INR
            embed_dim (int): size of one-scale pos embedding
            num_scales (int): number of scales in coord embedding
            dim (int): width of the INR
            depth (int): depth of the INR
        """

        super().__init__()

        self.dim   = dim
        self.depth = depth

        num_scales = len(scales)

        # input proj layer:
        layers = [nn.Linear(embed_dim * num_scales, dim), nn.ReLU()]

        # add intermediate layers based on depth:
        for _ in range(depth - 1):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        
        self.start_quantile  = start_quantile
        self.end_quantile    = end_quantile
        self.nb_quantiles    = nb_quantiles
        self.quantile_median = quantile_median

        # output layer:
        layers.append(nn.Linear(dim, nb_quantiles))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        localized_latents: torch.Tensor,
        return_act: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """ Input of size `(bs, seq_len, num_scales, dim_scale)`"""

        # stack the different scales:
        localized_latents = rearrange(localized_latents, "b n s c -> b n (s c)")

        if return_act:
            hidden_features = []
            for layer in self.mlp:
                localized_latents = layer(localized_latents)
                hidden_features.append(localized_latents)
            return localized_latents, hidden_features[:-1]

        else:

            # through the MLP:
            return self.mlp(localized_latents) # (bs, seq_len, output_dim)


# -------------------------------------
# NOT USED BUT LOOKS INTERESTING
# -------------------------------------

class AdaLN(nn.Module):
    def __init__(self, hidden_dim):
        super(AdaLN, self).__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc_scale = nn.Linear(hidden_dim, hidden_dim)
        self.fc_shift = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, z):
        # Apply LayerNorm first
        x_ln = self.ln(x)
        # Compute scale and shift parameters conditioned on z
        scale = self.fc_scale(z)  # .unsqueeze(1)
        shift = self.fc_shift(z)  # .unsqueeze(1)
        # Apply AdaLN transformation
        return scale * x_ln + shift


# Residual block class
class ModulationBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ModulationBlock, self).__init__()
        self.adaln1 = AdaLN(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.adaln2 = AdaLN(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, z):
        # Apply first AdaLN and linear transformation
        residual = x
        out = self.adaln1(x, z)
        out = self.silu(self.linear1(out))
        # Apply second AdaLN and linear transformation
        out = self.adaln2(out, z)
        out = self.linear2(out)
        # Residual connection
        return out + residual


# not used  but looks interesting
class LocalityAwareINRDecoderWithModulation(nn.Module):
    def __init__(self, hidden_dim=256, num_blocks=2):
        super(LocalityAwareINRDecoderWithModulation, self).__init__()
        # Stack residual blocks
        self.blocks = nn.ModuleList(
            [ModulationBlock(hidden_dim) for _ in range(num_blocks)]
        )

    def forward(self, x, z):
        # Pass through each residual block
        for block in self.blocks:
            x = block(x, z)
        return x

