import torch
import torch.nn as nn
from torch.nn import functional


mse_fn = torch.nn.MSELoss()

per_element_mse_fn = torch.nn.MSELoss(reduction="none")
per_element_huber_fn = torch.nn.HuberLoss(reduction="none")

def batch_mse_fn(
    x1: torch.Tensor,
    x2: torch.Tensor
) -> torch.Tensor:
    """Computes MSE between two batches of signals while preserving the batch
    dimension (per batch element MSE).
    Args:
        x1 (torch.Tensor): Shape (batch_size, *).
        x2 (torch.Tensor): Shape (batch_size, *).
    Returns:
        MSE tensor of shape (batch_size,).
    """

    per_element_mse = per_element_mse_fn(x1, x2)

    return per_element_mse.view(x1.shape[0], -1).mean(dim=1)

class SmoothPinballLossTorch(nn.Module):
    """
    Smoth version of the pinball loss function.

    Parameters
    ----------
    quantiles : torch.tensor
    alpha : int
        Smoothing rate.

    Attributes
    ----------
    self.pred : torch.tensor
        Predictions.
    self.target : torch.tensor
        Target to predict.
    self.quantiles : torch.tensor
    """
    def __init__(
        self,
        quantiles: float,
        alpha: float=0.01
    ) -> None:
        super(SmoothPinballLossTorch,self).__init__()
        self.pred = None
        self.targes = None
        self.quantiles = quantiles
        self.alpha = alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the loss for the given prediction.
        """
        error = target - pred
        q_error = self.quantiles * error
        beta = 1 / self.alpha
        soft_error = functional.softplus(-error, beta)

        losses = q_error + soft_error

        return losses

class Batch_SmoothPinballLossTorch(nn.Module):

    def __init__(self, quantiles: float) -> None:
        super(Batch_SmoothPinballLossTorch, self).__init__()

        self.pred = None
        self.targes = None
        self.quantiles = quantiles

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        per_element_smooth_pinball = SmoothPinballLossTorch(self.quantiles)(pred, target)

        return per_element_smooth_pinball.view(pred.shape[0], -1).mean(dim=1)


class MultiQuantileSmoothPinballLoss(nn.Module):
    """
    Computes the Smooth Pinball Loss for multi-quantile regression.

    This loss function is a differentiable approximation of the standard 
    Pinball Loss, utilizing a Softplus transformation to smooth the 
    non-differentiable point at zero. It is designed for models predicting 
    multiple quantiles simultaneously.

    Args:
        quantiles (torch.Tensor): A 1D tensor of shape (Q,) containing 
            the target quantiles (e.g., [0.05, 0.1, ..., 0.95]).
        alpha (float): Smoothing parameter. As alpha approaches 0, 
            the loss converges to the standard Pinball Loss. Default: 0.01.

    Shape:
        - pred: (B, T, Q) where B is batch size, T is sequence length, 
          and Q is the number of quantiles.
        - target: (B, T, 1) or (B, T), representing the ground truth values.
        - Output: scalar (mean loss) or (B,) if reduction is modified.

    Notes:
        The formula used is:
        L(q, error) = q * error + alpha * log(1 + exp(-error / alpha))
    """
    def __init__(self, quantiles: torch.Tensor, alpha: float = 0.01):
        super().__init__()
        # Register as buffer to handle device placement (CPU/GPU) automatically
        self.register_buffer('quantiles', quantiles)
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the smooth pinball loss.
        
        Args:
            pred (torch.Tensor): Predicted quantiles of shape (Batch, Time, Q).
            target (torch.Tensor): Ground truth values of shape (Batch, Time, 1).
        
        Returns:
            torch.Tensor: The mean loss over all dimensions.
        """
        # Ensure target is broadcastable across the quantile dimension (Q)
        if target.dim() < pred.dim():
            target = target.unsqueeze(-1)
            
        error = target - pred
        
        # Linear part of the pinball loss
        q_error = self.quantiles * error
        
        # Smooth approximation of the max(0, -error) term
        # Softplus(x) = log(1 + exp(x)). 
        # Here we scale by alpha to control the smoothness.
        soft_error = self.alpha * functional.softplus(-error / self.alpha)

        losses = q_error + soft_error
        
        return losses