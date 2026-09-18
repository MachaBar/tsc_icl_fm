from __future__ import annotations

from typing import Literal, Optional, get_args

import torch
import torch.nn as nn
from einops import repeat

from src.modules.aroma.blocks.attention import CrossAttention, FeedForward
from src.modules.aroma.blocks.utils import PreNormCross, PreNorm


PoolingType = Literal["mean", "max", "sum", "attention", "cls", "perceiver"]
POOLING_TYPES = get_args(PoolingType)


class SeriesPooler(nn.Module):
    """Pools per-series latent tokens Z_val (M [+extras] tokens) into one vector per series.

    Meant to consume the output of `UnivariatePerceiverEncoder`/`PerceiverEncoder`
    called with `return_latents=True` (block ①② latents, before the decoder
    cross-attn to target coordinates), so that a single row-representation per
    series can be fed to a set/table transformer such as
    `ICLearningClassification`.

    Two orthogonal ablation axes:

    1. `pooling` -- how the M tokens are reduced to one vector.
    2. `include_label_token` -- whether the series' class label is injected
       *early*, as an extra token appended to the M tokens before pooling,
       rather than added to the pooled vector inside the ICL head.

    Args:
    ----------
    pooling : {"mean", "max", "sum", "attention", "cls", "perceiver"}
        - "mean" / "max" / "sum": parameter-free reduction over the M tokens.
        - "attention": Pooling by Multi-head Attention (PMA, Set Transformer
          style) -- a single learnable query cross-attends over the M tokens.
        - "perceiver": same idea, but as a full Perceiver-style block --
          learnable query, pre-norm cross-attention with a residual, then a
          pre-norm feedforward with a residual. Mirrors how the encoder's own
          `cross_attend_blocks` are built, so the reduction has the same
          capacity as one encoder cross-attention stage rather than a bare
          attention op.
        - "cls": pick out a single dedicated token instead of reducing over M.
          Requires the *encoder* to have been built with `use_cls_token=True`,
          whose output has the CLS token appended as the last of the M(+1)
          positions. Mutually exclusive with `include_label_token` (both would
          claim the last position).

    d_model : int
        Dimension of the latent tokens (= dimension of the pooled output).

    num_classes : int, optional
        Required when `include_label_token=True`. The label embedding table
        holds `num_classes + 1` entries -- the extra one is the "unknown label"
        embedding used for query series, whose class is what we predict.

    include_label_token : bool, default=False
        If True, `forward` expects a `labels` argument and appends one label
        token to the M latent tokens before pooling, so the label is mixed into
        the series representation by the pooling operation itself (early
        fusion) rather than added to the pooled vector downstream (late fusion,
        what `ICLearningClassification` does on its own).

    heads, dim_head, dropout : int, int, float
        Cross-attention hyperArgs:, only used by "attention"/"perceiver".
    """

    UNKNOWN_LABEL = -1

    def __init__(
        self,
        pooling: PoolingType,
        d_model: int,
        num_classes: Optional[int] = None,
        include_label_token: bool = False,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        if pooling not in POOLING_TYPES:
            raise ValueError(f"Unknown pooling type {pooling!r}, expected one of {POOLING_TYPES}")

        if include_label_token:
            if num_classes is None:
                raise ValueError("`num_classes` is required when `include_label_token=True`")
            if pooling == "cls":
                raise ValueError(
                    "`pooling='cls'` and `include_label_token=True` are mutually exclusive: "
                    "both would claim the last token position."
                )

        self.pooling             = pooling
        self.d_model             = d_model
        self.include_label_token = include_label_token
        self.num_classes         = num_classes

        if include_label_token:
            # last row (index `num_classes`) is the "unknown label" embedding, used for query series:
            assert num_classes is not None
            self.label_encoder = nn.Embedding(num_classes + 1, d_model)
            self.unknown_index = num_classes

        if pooling == "attention":
            self.query = nn.Parameter(torch.randn(1, d_model) * 0.02)
            self.cross_attn = CrossAttention(
                query_dim = d_model,
                key_dim   = d_model,
                value_dim = d_model,
                heads     = heads,
                dim_head  = dim_head,
                dropout   = dropout,
            )
            self.norm = nn.LayerNorm(d_model)

        elif pooling == "perceiver":
            self.query = nn.Parameter(torch.randn(1, d_model) * 0.02)
            self.cross_attn = PreNormCross(
                d_model,
                CrossAttention(
                    query_dim = d_model,
                    key_dim   = d_model,
                    value_dim = d_model,
                    heads     = heads,
                    dim_head  = dim_head,
                    dropout   = dropout,
                ),
                k_dim = d_model,
                v_dim = d_model,
            )
            self.cross_ff = PreNorm(d_model, FeedForward(d_model))
            self.norm = nn.LayerNorm(d_model)

    def _append_label_token(self, Z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Append one label-embedding token to each series' M latent tokens.

        `labels` uses `SeriesPooler.UNKNOWN_LABEL` (-1) to mark query series
        whose class is unknown; those get the dedicated "unknown" embedding.
        """

        is_known = labels != self.UNKNOWN_LABEL
        safe_labels = torch.where(
            is_known,
            labels.long(),
            torch.full_like(labels.long(), self.unknown_index),
        )
        label_emb = self.label_encoder(safe_labels)          # (..., d_model)
        return torch.cat([Z, label_emb.unsqueeze(-2)], dim=-2)  # (..., M+1, d_model)

    def forward(self, Z: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
        ----------
        Z : Tensor of shape (..., M, d_model)
            Per-series latent tokens (any number of leading batch dims).
            When `pooling == "cls"`, the CLS token is expected to be the last
            of the M tokens, matching how the encoder appends it in
            `forward(..., return_latents=True)`.

        labels : Tensor of shape (...), optional
            Required when `include_label_token=True`: the integer class of each
            series, with `SeriesPooler.UNKNOWN_LABEL` (-1) for query series.
            Leading dims must match `Z`'s leading dims.

        Returns:
        -------
        Tensor of shape (..., d_model)
            One pooled vector per series.
        """

        if self.include_label_token:
            if labels is None:
                raise ValueError("`labels` is required when `include_label_token=True`")
            if labels.shape != Z.shape[:-2]:
                raise ValueError(
                    f"`labels` shape {tuple(labels.shape)} does not match `Z` leading dims "
                    f"{tuple(Z.shape[:-2])}"
                )
            Z = self._append_label_token(Z, labels)
        elif labels is not None:
            raise ValueError("`labels` was passed but `include_label_token=False`")

        if self.pooling == "mean":
            return Z.mean(dim=-2)

        if self.pooling == "max":
            return Z.max(dim=-2).values

        if self.pooling == "sum":
            return Z.sum(dim=-2)

        if self.pooling == "cls":
            return Z[..., -1, :]

        # learned reductions: a query token cross-attends over the M(+1) tokens.
        *batch_dims, M, d = Z.shape
        Z_flat = Z.reshape(-1, M, d)                                # (n, M, d)
        q = repeat(self.query, "1 d -> n 1 d", n=Z_flat.shape[0])   # (n, 1, d)

        if self.pooling == "attention":
            out = self.cross_attn(q, k=Z_flat, v=Z_flat)            # (n, 1, d)
        else:  # "perceiver": residual cross-attn + residual feedforward
            out = self.cross_attn(q, k=Z_flat, v=Z_flat) + q        # (n, 1, d)
            out = self.cross_ff(out) + out                          # (n, 1, d)

        out = self.norm(out.squeeze(1))                             # (n, d)
        return out.reshape(*batch_dims, d)

    def extra_repr(self) -> str:
        return (
            f"pooling={self.pooling!r}, d_model={self.d_model}, "
            f"include_label_token={self.include_label_token}"
        )