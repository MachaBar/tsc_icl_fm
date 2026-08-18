from typing import Sequence, Optional, Tuple

import torch
import torch.nn as nn

from einops import repeat, rearrange


from src.modules.aroma.aroma.encoder import NaiveFourierEncoder
from src.modules.icl_learning import ICLearning, ICLearningCovar, ICLearningCrossAttn


class FourierEncoderDecoderICL(nn.Module):

    def __init__(
        self,
        encoder: NaiveFourierEncoder,
        head: ICLearning | ICLearningCovar | ICLearningCrossAttn,
        apply_asinh_transform: bool = False,
        *args,
        **kwargs
    ):
        
        super().__init__()

        self.encoder = encoder
        assert isinstance(self.encoder, NaiveFourierEncoder)

        self.tf_icl = head
        assert isinstance(self.tf_icl, ICLearning) or isinstance(self.tf_icl, ICLearningCovar) or isinstance(self.tf_icl, ICLearningCrossAttn)

        self.to_tf_icl = nn.Linear(self.encoder.out_dim, self.tf_icl.d_model)
        self.apply_asinh_transform = apply_asinh_transform
    
    def _prepare_inputs(
        self,
        series: torch.Tensor,
        series_covar: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        
        if self.apply_asinh_transform:
            series = torch.asinh(series)
            if series_covar is not None:
                series_covar = torch.asinh(series_covar)
        
        return series, series_covar
    
    def _prepare_outputs(
        self,
        series: torch.Tensor
    ) -> torch.Tensor:
        
        if self.apply_asinh_transform:
            series = torch.sinh(series)
        
        return series
    
    def forward(
        self,
        series: torch.Tensor,
        coords: torch.Tensor,
        target_coords: torch.Tensor,
        series_covar: Optional[torch.Tensor] = None,
        coords_covar: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
        sample_posterior: bool = False,
        undo_asinh_transform: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        

        # prepare inputs (asinh transform, etc.):
        series, series_covar = self._prepare_inputs(series, series_covar)

        # get train len for TF_ICL:
        train_size = series.shape[1]

        # encode:
        localized_latents = self.encoder(
            target_coords    = target_coords,
        ) # (bs, seq_len, num_bandwidth, pos_embed_dim)

        # stack freq scales:
        localized_latents = rearrange(localized_latents, 'b t s d -> b t (s d)')

        # map to ICL d_model:
        localized_latents = self.to_tf_icl(localized_latents) # (bs, seq_len, d_model)

        # ICL Transformer head:
        out = self.tf_icl(
            R       = localized_latents, # (bs, seq_len, d_model)
            y_train = series,            # (bs, context_len, 1)
            y_cov   = series_covar
        )                                # --> (bs, seq_len - context_len, 1)

        if undo_asinh_transform:
            out = self._prepare_outputs(out) # (bs, seq_len - context_len, 1)
            if out.isnan().any():
                print('sinh output is NaN!, input all nan?', series.isnan().all(), localized_latents.isnan().any())

        return out, torch.empty(0)
