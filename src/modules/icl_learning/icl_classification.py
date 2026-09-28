from __future__ import annotations

from typing import Callable, Optional

from torch import nn, Tensor
import torch

from src.modules.icl_learning.encoders import Encoder


class ICLearningClassification(nn.Module):
    """Series-wise in-context learning for classification.
    
    Args:
    ----------
    d_model : int
    num_blocks : int
        Number of blocks used in the ICL encoder (MHA + RoPE).
    nhead : int
        Number of attention heads of the ICL encoder.
    dim_feedforward : int
        Dimension of the feedforward network of the ICL encoder.
    num_classes : int
        Number of classes. Assumed fixed across the whole training run
        (a single `nn.Embedding`/`nn.Linear` head sized to `num_classes`).
    dropout : float, default=0.0
        Dropout probability.
    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable.
    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward).
    inject_labels : bool, default=True
        Whether to add a learned label embedding onto the context rows of
        (late fusion). Set to False when labels were already injected upstream
        by a `SeriesPooler` built with `include_label_token=True
    """

    def __init__(
        self,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_classes: int,
        dropout: float = 0.0,
        activation: str | Callable[[Tensor], Tensor] = "gelu",
        norm_first: bool = True,
        inject_labels: bool = True,
    ):
        super().__init__()
        self.norm_first    = norm_first
        self.inject_labels = inject_labels

        self.tf_icl = Encoder(
            num_blocks      = num_blocks,
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = activation,
            norm_first      = norm_first,
        )
        if self.norm_first:
            self.ln = nn.LayerNorm(d_model)

        if inject_labels:
            self.label_encoder = nn.Embedding(num_classes, d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, num_classes)
        )

        self.num_classes = num_classes
        self.d_model     = d_model

    def _icl_predictions(self, R: Tensor, y_train: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """In-context learning predictions.

        Args:
        ----------
        R : Tensor
            Series representations of shape (B, N, D) where:
             - B is the number of "mini-datasets"
             - N is the number of series
             - D is the dimension of a series representation
        y_train : Tensor of shape (B, train_size)
            Integer class labels for the first `train_size` series (context),
            where `train_size` is the position to split `R` into context/query.
        key_padding_mask : Optional[Tensor of shape (B, N)], default=None
            True at a position = série fictive (padding, ex. classe absente de
            cet épisode dans un schéma num_classes fixé à MAX_CLASSES) --
            exclue des clés/valeurs de l'attention dans `self.tf_icl` (voir
            Encoder.forward, déjà supporté nativement). Les positions paddées
            de `y_train` (label arbitraire, jamais utilisé puisqu'exclues de
            l'attention) doivent rester dans [0, num_classes) côté appelant
            pour ne pas faire planter `self.label_encoder`.
        """

        train_size = y_train.shape[1]
        if self.inject_labels:
            R = R.clone()
            R[:, :train_size] = R[:, :train_size] + self.label_encoder(y_train.long())
        src = self.tf_icl(R, attn_mask=train_size, key_padding_mask=key_padding_mask)
        if self.norm_first:
            src = self.ln(src)
        out = self.decoder(src)  # (B, N, num_classes)

        return out

    def forward(self, R: Tensor, y_train: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """In-context classification based on learned series representations.

        Args:
        ----------
        R : Tensor
            Series representations of shape (B, N, D).
        y_train : Tensor of shape (B, train_size)
            Integer class labels for the context series.
        key_padding_mask : Optional[Tensor of shape (B, N)], default=None
            True = série fictive (padding), voir `_icl_predictions`. Les
            logits renvoyés pour ces positions ne doivent PAS être utilisés
            dans la loss côté appelant (voir aussi le masquage des classes
            absentes -- num_classes ici est fixé à MAX_CLASSES, donc les
            logits couvrent des classes qui peuvent ne pas exister pour cet
            épisode : à masquer séparément avant la cross-entropy, ce module
            ne connaît pas le n_classes réel de chaque épisode).


        Returns:
        -------
        Tensor
            Logits of shape (B, N - train_size, num_classes) for the query series.
        """

        train_size = y_train.shape[1]
        out = self._icl_predictions(R, y_train, key_padding_mask=key_padding_mask)
        out = out[:, train_size:]

        return out