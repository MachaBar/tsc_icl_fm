import logging
from collections import defaultdict

from typing import Dict, Union, Any
from pathlib import Path

from time import time
import pandas as pd

import torch
from torch.utils.data import DataLoader

from src.data.scaler import CustomStandardScaler
from src.modules.baselines.forecasting import LOCF, LastHorizonCF, SeasonalNaive
from src.tools.utils.plot import plot_forecasts
from src.tools.inference.common import update_results

def infer_fn(
    test_loader: DataLoader,
    path_results_exp: Path,
    horizon_len_in_days: int,
    nb_timesteps_per_day: int,
    naive_seasonality_in_days: int = 7,
    make_plots: bool = True,
    nb_plots: int = 3,
    max_points_to_plot: int = 335
) -> None:    
    """
    Main function for running TabPFN-TS at inference (with or without covariate).

    Parameters:
        ckpt_path (Path): Path to TabPFNRegressor ckpt
        test_loader (DataLoader): test dataloader
        horizon_len (int): actual horizon length to predict
        path_results_exp (Path): path where to export results
        quantile_levels (List[float]): list of quantile_levels for probabilistic head
        make_plots (bool): make forecasting plots or not
        nb_plots (int): number of samples to plot
        max_points_to_plot (int): restrict plot to the last `max_points_to_plot` timesteps
    """

    logger = logging.getLogger(__name__)

    infer_path = path_results_exp
    infer_path.mkdir(exist_ok=True)

    # get device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # instantiate local models:
    locf = LOCF(
        horizon_in_days      = horizon_len_in_days,
        nb_timesteps_per_day = nb_timesteps_per_day
    )

    lhcf = LastHorizonCF(
        horizon_in_days      = horizon_len_in_days,
        nb_timesteps_per_day = nb_timesteps_per_day
    )

    seasonal = SeasonalNaive(
        seasonality_in_days  = naive_seasonality_in_days,
        horizon_in_days      = horizon_len_in_days,
        nb_timesteps_per_day = nb_timesteps_per_day
    )

    horizon_len = horizon_len_in_days * nb_timesteps_per_day

    list_model_names = ['LOCF', 'LHCF', 'SeasonalNaive']
    list_models      = [locf, lhcf, seasonal]

    results: Dict[str, Dict[str, Any]]   = {k:defaultdict(list[torch.Tensor]) for k in list_model_names}
    materials: Dict[str, Dict[str, Any]] = {k:defaultdict(list[torch.Tensor]) for k in list_model_names}

    nb_chunks = 0

    t_start = time()

    # iterate through the test dataloader:
    for substep, batch in enumerate(test_loader):

        # 1. extract tensors from batch:
        series_c = batch.get('context_features').to(device)     # [N, T_in, 1] (context values)
        series_t = batch.get('target_features').to(device)      # [N, T_out, 1] (target values)
        coords_c = batch.get('context_coordinates').to(device)  # [N, T_in, 1] (context grid coordinates)
        coords_t = batch.get('target_coordinates').to(device)   # [N, T_out, 1] (target grid coordinates)
        mask     = batch.get('is_missing_mask').to(device)      # [N, T_out, 1] (where the samples are not observed)

        # 2. prepare data (normalize & filter):

        # z-normalize context series:
        scaler   = CustomStandardScaler(dim=1,epsilon=1e-7)
        scaler.fit(series_c)

        # get mask for flat windows:
        std_mask = (scaler.std > 1e-6).squeeze()    # [N,]
        series_c = scaler.transform(series_c)[std_mask]

        # z-normalize target series:
        series_t = scaler.transform(series_t)[std_mask] # [N', T_out, 1]
        
        # filter out flat contexts:
        coords_c, coords_t, mask = coords_c[std_mask], coords_t[std_mask], mask[std_mask]

        nb_chunks += len(series_t)

        yhat = dict()
        for model_name, model in zip(list_model_names, list_models):

            # 3. make predictions:
            yhat[model_name] = series_t.clone()
            yhat[model_name][:,-horizon_len:,:] = model(series_c)

            # 4. compute metrics on normalized data:
            results[model_name] = update_results(series_t, yhat[model_name], mask, results[model_name], is_normalized=True)

            if make_plots:
                materials[model_name]['coords_c'].append(coords_c.detach().cpu().squeeze())
                materials[model_name]['series_c'].append(series_c.detach().cpu().squeeze())
                materials[model_name]['coords_t'].append(coords_t.detach().cpu().squeeze())
                materials[model_name]['pred'].append(yhat[model_name].detach().cpu().squeeze())
                materials[model_name]['mask'].append(mask.cpu().squeeze())
                materials[model_name]['series_t'].append(series_t.detach().cpu().squeeze())

        # 5. compute metrics on raw data:

        # undo z-normalization (back to data space) and compute metrics:
        series_t = scaler.inv_transform(series_t, std_mask) # [N', T_out, 1]
        for model_name in list_model_names:
            yhat_               = scaler.inv_transform(yhat[model_name].to(device), std_mask)     # [N', T_out, 1]        
            results[model_name] = update_results(series_t, yhat_, mask, results[model_name], is_normalized = False)
    
    # end of tinference loop:
    t_end = time()

    # agregate all results:
    results_agg = {
        k: {
            key:torch.cat(val,dim=0).cpu().detach() for key,val in results[k].items() if len(val)>0
        }
        for k in list_model_names
    }
    results_agg = {
        k: {
            key:val.nanmean().cpu().detach().item() if ('mae' in key or 'mse' in key) and len(val)>0 else val for key,val in results_agg[k].items()
        }
        for k in list_model_names
    }

    save_dir_metrics = infer_path / 'metrics'
    save_dir_metrics.mkdir(exist_ok=True)

    # export metrics (MAE):
    df = pd.DataFrame({
        'Model': list_model_names,
        'Chunks': [nb_chunks]*len(list_model_names),
        'MAE on Missing Values' : [results_agg[k].get('mae_on_missing_values') for k in list_model_names],
        'MAE on Observed Values': [results_agg[k].get('mae_on_observed_values') for k in list_model_names],
        'MAE on Missing Values (norm)' : [results_agg[k].get('mae_on_missing_values_norm') for k in list_model_names],
        'MAE on Observed Values (norm)': [results_agg[k].get('mae_on_observed_values_norm') for k in list_model_names],
        'Time (s)': [t_end - t_start]*len(list_model_names)
    })
    df.set_index('Model', inplace=True)
    df.to_csv(save_dir_metrics / 'mae.csv')

    # export metrics (MSE):
    df = pd.DataFrame({
        'Model': list_model_names,
        'Chunks': [nb_chunks]*len(list_model_names),
        'MSE on Missing Values' : [results_agg[k].get('mse_on_missing_values') for k in list_model_names],
        'MSE on Observed Values': [results_agg[k].get('mse_on_observed_values') for k in list_model_names],
        'MSE on Missing Values (norm)' : [results_agg[k].get('mse_on_missing_values_norm') for k in list_model_names],
        'MSE on Observed Values (norm)': [results_agg[k].get('mse_on_observed_values_norm') for k in list_model_names],
        'Time (s)': [t_end - t_start]*len(list_model_names)
    })
    df.set_index('Model', inplace=True)
    df.to_csv(save_dir_metrics / 'mse.csv')
    
    # concat batches outputs
    for model_name in list_model_names:
        materials[model_name] = {key:torch.cat(val,dim=0) for key,val in materials[model_name].items() if len(val)>0}

        if make_plots:
            plot_forecasts(
                materials  = materials[model_name],
                nb_plots   = min(3, nb_plots),
                save_dir   = infer_path,
                gt_exists  = True,
                max_points = max_points_to_plot,
                model_name = model_name,
                show_context_points = True,
                plot_covar = False
            )

    return
