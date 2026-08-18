import torch
import torch.nn as nn

from src.modules.timeflow.modulation import film, film_linear, film_translate
from src.modules.timeflow.hypernetwork import LatentToModulation

from ._base import FourierFeatures

class ModulatedFourierFeatures(FourierFeatures):

    """WARNING: the code does not support non-graph inputs.
        It needs to be adapted for (batch, num_points, coordinates) format
        The FiLM Modulated Network with Fourier Embedding used for the experiments on Airfrans.
        The code relies on conditoning functions: film, film_linear and film_translate.
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        num_frequencies: int = 8,
        latent_dim: int = 128,
        width: int = 256,
        depth: int = 5,
        modulate_scale: bool = False,
        modulate_shift: bool = True,
        frequency_embedding: str = "nerf",
        include_input: bool = True,
        scale: int = 5,
        log_sampling: bool = False,
        min_frequencies: int = 0,
        max_frequencies: int = 32,
        base_frequency: float = 1.25,
    ) -> None:
        
        super().__init__(
            input_dim           = input_dim,
            output_dim          = output_dim,
            num_frequencies     = num_frequencies,
            width               = width,
            depth               = depth,
            frequency_embedding = frequency_embedding,
            include_input       = include_input,
            scale               = scale,
            log_sampling        = log_sampling,
            min_frequencies     = min_frequencies,
            max_frequencies     = max_frequencies,
            base_frequency      = base_frequency
        )

        self.latent_dim = latent_dim  # dim of the latent code 

        # number of modulation parameters:
        self.num_modulations = self.hidden_dim * (self.depth - 1)  
        if modulate_scale and modulate_shift:
            self.num_modulations *= 2

        # hypernetwork that outputs the modulations:
        self.latent_to_modulation = LatentToModulation(
            self.latent_dim,
            self.num_modulations,
            dim_hidden=256, # XXX
            num_layers=1    # XXX
        )

        # define modulation mechanism:
        if modulate_shift and modulate_scale:
            self.conditioning = film

        elif modulate_scale and not modulate_shift:
            self.conditioning = film_linear

        else:
            self.conditioning = film_translate

    def modulated_forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:

        # x -> tensor of (time) coordinates [*, T, input_dim]
        # z -> tensor of latent codes [..., latent_dim]

        # decode latent code into modulation params:
        features = self.latent_to_modulation(z) # [..., num_modulations]

        # encode (time) coordinates:
        position = self.embedding(x) # [*, T, hdim] with hdim = encoding.out_dim or hdim = encoding.out_dim + input_dim

        # handle Gaussian encoding of coords with include input:
        if self.frequency_embedding == "gaussian" and self.include_input:
            position = torch.cat([position, x], dim=-1) # [*, hdim] with hdim = encoding.out_dim + input_dim

        # apply modulation to (all but the last) layers of the INR:
        pre_out = self.conditioning(position, features, self.layers[:-1], torch.relu)

        # apply final layer:
        out = self.layers[-1](pre_out) # [*, T, output_dim]

        return out # [*, T, output_dim]
    

