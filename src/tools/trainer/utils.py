from typing import Tuple, Dict
import torch

from src.data.scaler import CustomStandardScaler
from src.tools.utils.mask import std_mask


def unpack_batch(
    batch: dict,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    
    # get context (observed values):
    series_c = batch.get('context_features').to(device, non_blocking=True)    # [N, T_in, 1] (context values)
    coords_c = batch.get('context_coordinates').to(device, non_blocking=True) # [N, T_in, 1] (context grid coordinates)

    # get target (missing instants only):
    series_t = batch.get('missing_features').to(device, non_blocking=True)    # [N, T_mis, 1] (missing values)
    coords_t = batch.get('missing_coordinates').to(device, non_blocking=True) # [N, T_mis, 1] (missing grid coordinates)

    # for some samples, context is full of NaNs --> discard:
    no_nan_check = torch.isnan(series_c).sum(1).squeeze() == 0

    coords_c, series_c = coords_c[no_nan_check], series_c[no_nan_check]
    coords_t, series_t = coords_t[no_nan_check], series_t[no_nan_check]

    # get covar (same grid length as context):
    covar_c = batch.get('context_covar')    # [N, c, T_in, 1] 
    covar_t = batch.get('missing_covar')    # [N, c, T_mis, 1] 
    # if covar_c is None:
    #     covar_c = torch.zeros_like(series_c).unsqueeze(1)
    # series_cov = batch.get('covar_values')
    # coords_cov = batch.get('covar_coordinates')


    has_covar   = covar_c is not None #(series_cov is not None) and (coords_cov is not None)
    has_covar_t = covar_t is not None
    if has_covar:
        # series_cov = series_cov.to(device, non_blocking=True)
        # coords_cov = coords_cov.to(device, non_blocking=True)
        covar_c = covar_c.to(device, non_blocking=True)[no_nan_check]
    if has_covar_t:
        covar_t = covar_t.to(device, non_blocking=True)[no_nan_check]

    return series_c, coords_c, series_t, coords_t, covar_c, covar_t


def znorm_fit_transform(
    series_c: torch.Tensor,
    coords_c: torch.Tensor,
    series_t: torch.Tensor,
    coords_t: torch.Tensor,
    covar_c: torch.Tensor | None,
    covar_t: torch.Tensor | None,
    eps: float = 1e-5,
    max_znorm: float = 20,
    apply_std_mask: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    
    # znormalize target series
    scaler = CustomStandardScaler(dim=1, epsilon=eps)
    scaler.fit(series_c)
    series_c = scaler.transform(series_c)   # (bs, len_context, 1)
    series_t = scaler.transform(series_t)   # (bs, len_target, 1)

    covar_all = None
    if covar_c is not None:
        covar_all = torch.cat([covar_c, covar_t], dim=2) if (covar_t is not None) else covar_c

        scaler_cov = CustomStandardScaler(dim=2, epsilon=eps)
        scaler_cov.fit(covar_all)
        covar_all = scaler_cov.transform(covar_all)     # (bs, c, len_covar, 1)

        # series_cov, coords_cov = series_cov[no_nan_check], coords_cov[no_nan_check]
        # scaler_cov = CustomStandardScaler(dim=1, epsilon=1e-5)
        # scaler_cov.fit(series_cov)
        # series_cov = scaler_cov.transform(series_cov)
        # series_cov, coords_cov = series_cov[values_mask], coords_cov[values_mask]
        
    # remove weird samples from series:
    if max_znorm > 0:
        values_mask = (series_t.max(1)[0] < max_znorm).squeeze() # change here for removing more XXX
        series_c, series_t = series_c[values_mask], series_t[values_mask]
        coords_c, coords_t = coords_c[values_mask], coords_t[values_mask]
        if covar_all is not None:
            covar_all = covar_all[values_mask]
    
    # If apply_mask_during_training, remove samples with very small standard deviation:
    if apply_std_mask:

        _, mask_c = std_mask(series_c, threshold=8.0, return_mask=True)
        _, mask_t = std_mask(series_t, threshold=8.0, return_mask=True)
        mask_std  = torch.logical_and(mask_c, mask_t)
        
        series_c, coords_c = series_c[mask_std], coords_c[mask_std]
        series_t, coords_t = series_t[mask_std], coords_t[mask_std]

        if (covar_all is not None):
            covar_all = covar_all[mask_std]
            # series_cov, coords_cov = series_cov[mask_std], coords_cov[mask_std]

        del mask_std, mask_c, mask_t
    
    return series_c, coords_c, series_t, coords_t, covar_all


def update_metrics(
    yhat: torch.Tensor,
    series: torch.Tensor,
    metric_epoch: Dict[str, float],
    per_step_callback,
    key: str = 'mae_target'
):
    
    n_samples = series.shape[0]

    metric= torch.nn.MSELoss() if 'mse' in key else torch.nn.L1Loss()

    with torch.no_grad():
        
        # get MAE/MSE on unobserved coords only:
        loss_target_mae    = metric(yhat, series).cpu().detach()
        metric_epoch[key] += loss_target_mae.item() * n_samples

    per_step_callback[key].append(loss_target_mae)

    return per_step_callback, metric_epoch


# with torch.no_grad():
    
#     if yhat_context is not None:
#         yhat_context = yhat_context[...,median_idx:median_idx+1]
#     yhat_target = yhat_target[..., median_idx:median_idx+1]

#     # optionnally, undo asinh transform to get metrics in z-norm space:
#     series_c = inv_transform(series_c)
#     if yhat_context is not None:
#         yhat_context      = inv_transform(yhat_context)
#     series_t, yhat_target = inv_transform(series_t), inv_transform(yhat_target)

#     # get MAE/MSE on context grid only:
#     if yhat_context is not None:
#         loss_context_mae   = torch.nn.L1Loss(reduction='median')(yhat_context, series_c).cpu().detach()
#         mae_context_epoch += loss_context_mae.item() * n_samples
#         if track_mse:
#             loss_context_mse   = torch.nn.MSELoss()(yhat_context, series_c).cpu().detach()
#             mse_context_epoch += loss_context_mse.item() * n_samples
#     else:
#         mae_context_epoch, mse_context_epoch = 0.0, 0.0

#     # get MAE/MSE on unobserved coords only:
#     loss_target_mae   = torch.nn.L1Loss()(yhat_target, series_t).cpu().detach()
#     mae_target_epoch += loss_target_mae.item() * n_samples
#     if track_mse:
#         loss_target_mse   = torch.nn.MSELoss()(yhat_target, series_t).cpu().detach()
#         mse_target_epoch += loss_target_mse.item() * n_samples