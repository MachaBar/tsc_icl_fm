from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from src.modules.aroma.aroma.encoder import PerceiverEncoder, UnivariatePerceiverEncoder
from src.modules.aroma.inr import LocalityAwareINRDecoder

class AROMAEncoderDecoderKL(nn.Module):

    def __init__(
        self,
        encoder: PerceiverEncoder | UnivariatePerceiverEncoder,
        head: LocalityAwareINRDecoder,
        apply_asinh_transform: bool = True,
        *args,
        **kwargs
    ):
        """
        AROMA encoder-decoder: Perceiver architecture + INR decoder.

        Args:
            input_dim (int): number of coordinates
            num_channels (int): number of channels in values tensors
            depth_inr (inr): depth of the INR decoder
            dim (int): width of the INR decoder
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

        self.decoder = head
        assert isinstance(self.decoder, LocalityAwareINRDecoder)

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
        mask: Optional[torch.Tensor] = None,
        target_coords: Optional[torch.Tensor] = None,
        series_covar: Optional[torch.Tensor] = None,
        coords_covar: Optional[torch.Tensor] = None,
        return_stats: bool = False,
        sample_posterior: bool = True,
        return_act: bool = False,
        undo_asinh_transform: bool = True
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # prepare inputs (asinh transform, etc.):
        series, series_covar = self._prepare_inputs(series, series_covar)

        # encode:
        out = self.encoder(
            series           = series,
            coords           = coords,
            series_covar     = series_covar if series_covar is not None else None,
            coords_covar     = coords_covar if coords_covar is not None else None,    
            mask             = mask,
            target_coords    = target_coords,
            return_stats     = return_stats,
            sample_posterior = sample_posterior,
        )
        
        if return_stats:
            localized_latents, kl_loss, mean, logvar = out
        else:
            localized_latents, kl_loss = out
        # localized_latents -> (bs, target_seq_len, num_bandwidth, pos_embed_dim)

        # 
        output_features = self.decoder(localized_latents, return_act = return_act) # (bs, target_seq_len, num_channels)

        if undo_asinh_transform:
            output_features = self._prepare_outputs(output_features) # (bs, seq_len - context_len, 1)
        
        if return_act:
            return output_features

        if return_stats:
            return output_features, kl_loss, mean, logvar

        return output_features, kl_loss
