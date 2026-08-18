from typing import Optional, Tuple

import torch
import torch.nn as nn

from einops import rearrange

from src.modules.aroma.aroma.encoder import PerceiverEncoder, UnivariatePerceiverEncoder
from src.modules.icl_learning import ICLearning, ICLearningCovar, ICLearningCrossAttn


class AROMAEncoderDecoderICL(nn.Module):

    def __init__(
        self,
        encoder: PerceiverEncoder | UnivariatePerceiverEncoder,
        head: ICLearning | ICLearningCovar | ICLearningCrossAttn,
        apply_asinh_transform: bool = False,
        *args,
        **kwargs
    ):
        """
        AROMA encoder-decoder: Perceiver architecture + Transformer ICL decoder.

        Args:
            input_dim (int): number of coordinates
            num_channels (int): number of channels in values tensors
            num_latents (int): number of query tokens (*encode*)
            hidden_dim: (int): dim of the query tokens (*encode*)
            latent_dim: (int): dim of the tokens in latent space (*process*)
            num_self_attentions (int): number of SA layers in latent space (*process*)
            latent_heads (int): number of latent self-attn heads (*process*)
            latent_dim_head (int): dim of each latent self-attn head (*process*)
            bottleneck_index (int): where to compress to `latent dim` wrt to the self-attn layers (*process*)
            cross_heads: (int): number of cross-attn heads (*encode* and *decode*)
            cross_dim_head (int): dim of each cross-attn head (*encode* and *decode*)
            max_pos_encoding_freq (int): max frequ (log2 scale) for Fourier feature pos embeddings of the geo encoder (*encode*)
            num_freq (int): number of frequ bands to sample for Fourier features (*encode* and *decode*)
            scales (Sequence[int]): in log2 scale, Fourier features bounds of each band (multi-band, *decode*)
            mlp_feature_dim: (int): feature dimensions before the MLP (*decode*)
            decoder_ff (bool): optional 2-layer MLP after mutli-scale cross-attn (*decode*)
            encode_geo (bool): whether to encode the geomeotry (*encode*)
            include_pos_in_value (bool): whether to include coordinates in the values of the value (pixel) encoder (*encode*)            
        """
        
        super().__init__()

        self.encoder = encoder
        assert isinstance(self.encoder, PerceiverEncoder) or isinstance(self.encoder, UnivariatePerceiverEncoder)

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
        out = self.encoder(
                series           = series,
                coords           = coords,
                series_covar     = series_covar[..., train_size:, :] if series_covar is not None else None,
                coords_covar     = coords_covar[..., train_size:, :] if coords_covar is not None else None,
                mask             = mask,
                target_coords    = target_coords,
                return_stats     = return_stats,
                sample_posterior = sample_posterior,
            )
        
        if return_stats:
            localized_latents, kl_loss, mean, logvar = out
        else:
            localized_latents, kl_loss = out
        # localized_latents -> (bs, seq_len, num_bandwidth, pos_embed_dim)

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

        if return_stats:
            return out, kl_loss, mean, logvar

        return out, kl_loss