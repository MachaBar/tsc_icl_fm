from typing import Dict, Union, List, Optional

import torch
import torch.nn as nn


class FourierTokenizer:

    def __init__(
        self,
        max_grid_len_in_days: int,
        nb_timesteps_per_day: int,
        list_freq_in_days: List[Union[int, float]] = [1, 7]
    ) -> None:
        """
        Args:
            max_grid_len_in_days (int): number of timesteps in the full-length grid
            nb_timesteps_per_day (int): dataset-specific number of timesteps in one day
            list_freq_in_days (List[Union[int, float]]): desired periods expressed in days
        """
        
        self.max_seq_len   = max_grid_len_in_days
        self.seasonalities = [f * nb_timesteps_per_day for f in list_freq_in_days]
    
    def _build_features(
        self,
        coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        """ Get Fourier features for tensor of shape `[N, T, 1]`."""

        if coords is None:
            return

        fourier_features  = [coords.clone()]
        fourier_features += [
            fn( 2 * torch.pi * self.max_seq_len * coords / season )
            for season in self.seasonalities
            for fn in [torch.sin, torch.cos]
        ] # 5x[N, T, 1]
        
        return torch.stack( fourier_features, dim = -1 ) # [N, T, 1, 5]

    def __call__(
        self,
        coords_c: torch.Tensor,
        coords_t: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor | None]:
        """
        Args:
            coords_c (torch.Tensor): batch of context coordinates (*, T, 1)
            series_c (torch.Tensor): batch of context values (*, T, 1)
            coords_t (torch.Tensor, optional): batch of target coordinates (*, T', 1)
        """
        
        H_context = self._build_features(coords_c) # [*, T, 1, 5]
        H_target  = self._build_features(coords_t) if coords_t is not None else None # [*, T, 1, 5]
        
        output = {
            'H_context' : H_context,
            'H_target'  : H_target,
        }

        return output
    
    # def forward(
    #     self,
    #     coords_c: torch.Tensor,
    #     coords_t: Optional[torch.Tensor] = None
    # ) -> Dict[str, torch.Tensor | None]:
    #     """
    #     Args:
    #         coords_c (torch.Tensor): batch of context coordinates (*, T, 1)
    #         series_c (torch.Tensor): batch of context values (*, T, 1)
    #         coords_t (torch.Tensor, optional): batch of target coordinates (*, T', 1)
    #     """
        
    #     H_context = self._build_features(coords_c) # [*, T, 1, 5]
    #     H_target  = self._build_features(coords_t) # [*, T, 1, 5]
        
    #     output = {
    #         'H_context'     : H_context,
    #         'H_target'      : H_target,
    #     }

    #     return output