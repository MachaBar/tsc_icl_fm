from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch

from gluonts.ev.metrics import MAE, MSE, RMSE, NRMSE, MeanWeightedSumQuantileLoss, DirectMetric
# MAPE, MASE, MSIS, ND, SMAPE,

def initialize_gluonts_metrics(
    axis: int | None = 1
) -> Dict[str, DirectMetric]:
    """
    Initialize list of GluonTS metrics to compute at inference.
    """
    
    metrics = (
        MSE(forecast_type="mean"),
        MSE(forecast_type='0.5'),
        MAE(forecast_type="mean"),
        MAE(forecast_type="0.5"),
        RMSE(),
        NRMSE(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    )

    evaluators = {}
    for metric in metrics:
        evaluator = metric(axis=axis)
        evaluators[evaluator.name] = evaluator

    return evaluators

def update_gluonts_metrics(
    ytrue: torch.Tensor,
    yhat: torch.Tensor,
    evaluators: Dict[str, DirectMetric],
    is_target_mask: Optional[torch.Tensor] = None
) -> Dict[str, DirectMetric]:
    """
    Update metrics with GluonTS submodule.

    Args:
        ytrue (torch.Tensor): ground truth of shape `(n, T, 1)`
        yhat (torch.Tensor): forecasts of shape `(n, T, q)`
        evaluators (Dict[str, DirectMetric]): dict of GluonTS metrics
        is_target_mask (torch.Tensor): boolean mask of shape `(n, T, 1)`, True for unobserved time indices
    
    Returns:
        Same `evaluators` with updated metrics
    """
    
    # target mask:
    mask = torch.ones(len(ytrue)).bool() if is_target_mask is None else is_target_mask
    mask.to(ytrue.device)

    # prepare groud truth and mean pred on target:
    label = ytrue[mask]
    mean  = yhat.mean(dim=-1, keepdim=True)[mask]

    # remove unobserved GT:
    nan_mask = ~label.isnan()
    label = label[nan_mask]
    mean  = mean[nan_mask]

    # instantiate batch dict:
    batch = {
        'label' : label.cpu().detach().numpy(),
        'mean'  : mean.cpu().detach().numpy(),
    }
    
    # get 10th, 20th, ... quantiles:
    if yhat.shape[2] > 1:
        inc  = int(0.1 * (yhat.shape[2] +1))
        yhat = yhat[...,inc-1::inc]
        assert yhat.shape[2] == 9

    # update batch dict:
    for q in range(9):
        idx = q * (yhat.shape[2]>1)
        yhat_q = yhat[...,idx:idx+1][mask]
        yhat_q = yhat_q[nan_mask]
        batch[str((q+1)/10)] = yhat_q.cpu().detach().numpy()

    # evaluate:
    for evaluator in evaluators.values():
        evaluator.update(batch)

    return evaluators

def update_results(
    ytrue: torch.Tensor,
    yhat: torch.Tensor,
    mask: torch.Tensor,
    results: Dict[str, List[torch.Tensor]],
    is_normalized: bool,
    model_name: str = ''
) ->  Dict[str, List[torch.Tensor]]:

    error  = torch.abs(yhat - ytrue)          # [N', T_out, 1]    
    mae_on_missing_values  = error[mask]      # Forecasting -> MAE on horizon
    mae_on_observed_values = error[~mask]     # Forecasting -> MAE on lookback
    mse_on_missing_values  = error[mask]**2   # Forecasting -> MSE on lookback
    mse_on_observed_values = error[~mask]**2  # Forecasting -> MSE on horizon

    suffix = '_norm' if is_normalized else ''
    suffix += '_' if len(model_name)>0 else ''

    if len(mae_on_missing_values) > 0:
        results['mae_on_missing_values'+suffix+model_name].append(mae_on_missing_values.detach().cpu())
        results['mse_on_missing_values'+suffix+model_name].append(mse_on_missing_values.detach().cpu())
    if len(mae_on_observed_values) >0:
        results['mae_on_observed_values'+suffix+model_name].append(mae_on_observed_values.detach().cpu())
        results['mse_on_observed_values'+suffix+model_name].append(mse_on_observed_values.detach().cpu())

    return results

def get_mae_mse(
    ground_truth: torch.Tensor,
    is_target_mask: torch.BoolTensor,
    list_model_forecasts: List[torch.Tensor],
    list_model_names: Optional[List[str]] = None,
    is_z_normalized: bool = False,
    results: Optional[Dict[str, List[torch.Tensor]]] = None,
    materials: Optional[Dict[str, List[torch.Tensor]]] = None
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
    
    if list_model_names is None:
        list_model_names = ['Model {}'.format(i+1) for i in range(len(list_model_forecasts))]
    
    if results is None:
        results = defaultdict(list)
    
    if materials is None:
        materials = defaultdict(list)

    assert ground_truth.shape == is_target_mask.shape
    assert len(list_model_forecasts) == len(list_model_names)
    assert ground_truth.ndim == 3
    assert ground_truth.shape[-1] == 1

    suffix = '_norm' if is_z_normalized else ''

    for forecast, model_name in zip(list_model_forecasts, list_model_names):

        results = update_results(ground_truth, forecast, is_target_mask, results, is_z_normalized, model_name)

        materials['forecast{}_{}'.format(suffix,model_name)].append(forecast.detach().cpu().squeeze())

    return results, materials