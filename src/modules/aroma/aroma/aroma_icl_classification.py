from typing import Optional, Tuple

import torch
import torch.nn as nn

from einops import rearrange

from src.modules.aroma.aroma.encoder import PerceiverEncoder, UnivariatePerceiverEncoder
from src.modules.icl_learning import ICLearningClassification, SeriesPooler


class EncoderICLClassifier(nn.Module):
    """Encoder (block 1 output Z_val) + kind of pooling + Transformer ICL classifier.

    Instead of querying the encoder's latents at each timestep to build a
    row-per-timestep sequence, pool the encoder's block latents
    (`Z_val`, before the decoder cross-attn to target coordinates) into a
    single vector per *series*, and feeds a sequence of series
    representations to `ICLearningClassification`: one row per series,
    context rows carry a class label, query rows do not.

    Args:
        encoder : PerceiverEncoder | UnivariatePerceiverEncoder
             `use_cls_token=True` at construction time if `pooler` uses `pooling="cls"`.
        pooler : SeriesPooler
            Reduces `Z_val`'s M(+1) latent tokens to one vector per series.
            If it was built with `include_label_token=True`, labels are injected *early* 
            as an extra token pooled together with the M latent tokens
            `inject_labels=False` then
        head : ICLearningClassification
            Transformer head performing in-context classification over series
        apply_asinh_transform : bool, default=False
            Whether to apply `asinh` to the raw values before encoding (mirrors
            `AROMAEncoderDecoderICL`)
    """

    def __init__(
        self,
        encoder: PerceiverEncoder | UnivariatePerceiverEncoder,
        pooler: SeriesPooler,
        head: ICLearningClassification,
        apply_asinh_transform: bool = False,
    ):
        super().__init__()

        self.encoder = encoder
        assert isinstance(self.encoder, (PerceiverEncoder, UnivariatePerceiverEncoder))

        self.pooler = pooler
        assert isinstance(self.pooler, SeriesPooler)

        self.head = head
        assert isinstance(self.head, ICLearningClassification)

        self.to_tf_icl = nn.Linear(self.encoder.latent_out_dim, self.head.d_model)
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

    def encode_series(
        self,
        series: torch.Tensor,
        coords: torch.Tensor,
        series_covar: Optional[torch.Tensor] = None,
        coords_covar: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        sample_posterior: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of *individual* series into their block ①② latent tokens.

        Args:
            series : Tensor of shape (bs, T, 1)
            coords : Tensor of shape (bs, T, 1)
            series_covar : Tensor of shape (bs, C-1, T, 1), optional
            coords_covar : Tensor of shape (bs, C-1, T, 1), optional
            mask : Tensor, optional
            sample_posterior : bool, default=False
                Whether to sample from the encoder's (optional) VAE bottleneck or
                use its mode (deterministic) -- see `use_kl` on `encoder`.

        Output:
            Z : Tensor of shape (bs, M[+cls], hidden_dim)
                Per-series latent tokens, not yet pooled (pooling needs the labels
                when `pooler.include_label_token`, so it happens in `forward`).
            kl_loss : Tensor (scalar)
        """

        series, series_covar = self._prepare_inputs(series, series_covar)

        Z, kl_loss = self.encoder(
            series           = series,
            coords           = coords,
            series_covar     = series_covar,
            coords_covar     = coords_covar,
            mask             = mask,
            return_latents   = True, # arrêter après le bloc 1 de l'encodeur pour la classif puis faire "le pooling"
            sample_posterior = sample_posterior,
        )
        # Z: (bs, M[+cls], hidden_dim)     for UnivariatePerceiverEncoder
        #    (bs, C, M[+cls], hidden_dim)  for PerceiverEncoder

        if Z.ndim == 4:
            Z = Z[:, 0]  # target-series channel only (covariate channels not used here)

        return Z, kl_loss

    def forward(
        self,
        series: torch.Tensor,
        coords: torch.Tensor,
        y_train: torch.Tensor,
        series_covar: Optional[torch.Tensor] = None,
        coords_covar: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        sample_posterior: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            series : Tensor of shape (bs, N, T, 1)
                `N` series (context + query) per episode, each of length `T`
                (series in one episode are assumed to share the same grid length;
                pad/crop upstream if they don't).
            coords : Tensor of shape (bs, N, T, 1)
            y_train : Tensor of shape (bs, train_size)
                Integer class labels for the first `train_size` series (context)
                of each episode.
            series_covar : Tensor of shape (bs, N, C-1, T, 1), optional
            coords_covar : Tensor of shape (bs, N, C-1, T, 1), optional
            mask : Tensor, optional
            sample_posterior : bool, default=False

        Output:
            logits : Tensor of shape (bs, N - train_size, num_classes)
            kl_loss : Tensor (scalar)
        """

        bs, N, T, _ = series.shape

        series_flat = rearrange(series, 'b n t 1 -> (b n) t 1')
        coords_flat = rearrange(coords, 'b n t 1 -> (b n) t 1')

        series_covar_flat, coords_covar_flat = None, None
        if series_covar is not None:
            assert coords_covar is not None
            series_covar_flat = rearrange(series_covar, 'b n c t 1 -> (b n) c t 1')
            coords_covar_flat = rearrange(coords_covar, 'b n c t 1 -> (b n) c t 1')

        Z, kl_loss = self.encode_series(
            series           = series_flat,
            coords           = coords_flat,
            series_covar     = series_covar_flat,
            coords_covar     = coords_covar_flat,
            mask             = mask,
            sample_posterior = sample_posterior,
        )                                                    # ((b n), M, hidden_dim)

        Z = rearrange(Z, '(b n) m d -> b n m d', b=bs, n=N)   # (bs, N, M, hidden_dim)

        pooler_labels = None
        if self.pooler.include_label_token:
            # context series carry their label, query series the "unknown" marker:
            train_size = y_train.shape[1]
            pooler_labels = torch.full(
                (bs, N), SeriesPooler.UNKNOWN_LABEL, dtype=torch.long, device=Z.device
            )
            pooler_labels[:, :train_size] = y_train.long()

        R = self.pooler(Z, labels=pooler_labels)             # (bs, N, hidden_dim)
        R = self.to_tf_icl(R)                                # (bs, N, d_model)

        logits = self.head(R, y_train)                       # (bs, N - train_size, num_classes)

        return logits, kl_loss