import logging
from collections import defaultdict

from pathlib import Path

from time import time
from typing import Literal
import pandas as pd

import torch
from torch.utils.data import DataLoader

from src.data.scaler import CustomStandardScaler
from src.modules.baselines.imputation import LinearInterpolation, CubicSplineInteporlation, SeasonalNaive
from src.tools.inference.common import initialize_gluonts_metrics, update_gluonts_metrics

def infer_fn(
    baseline: Literal['linear', 'seasonal', 'spline'],
    test_loader: DataLoader,
    path_results_exp: Path,
    nb_timesteps_per_day: int,
    naive_seasonality_in_days: int = 1,
    make_plots: bool = True,
    nb_plots: int = 3,
    max_points_to_plot: int = 672
) -> None:    
    """
    Main function for running Local imputation baselines at inference (with or without covariate).
    """

    logger = logging.getLogger(__name__)

    infer_path = path_results_exp / 'inference'
    infer_path.mkdir(exist_ok=True)

    # get device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_models = ['Linear', 'Spline', 'Seasonal'] 
    dict_evaluators = {k:initialize_gluonts_metrics(axis=None) for k in all_models}

    # (i) instantiate linear interpolation:
    linear   = LinearInterpolation()

    # (ii) instantiate cubic spline interpolation:
    spline   = CubicSplineInteporlation()

    # (ii) instantiate seasonal naive baseline:
    seasonal = SeasonalNaive(
        seasonality_in_days  = naive_seasonality_in_days,
        nb_timesteps_per_day = nb_timesteps_per_day
    )

    if baseline == 'linear':
        list_model_names = ['Linear']
        list_models = [linear]
    elif baseline == 'seasonal':
        list_model_names = ['Seasonal']
        list_models = [seasonal]
    elif baseline == 'spline':
        list_model_names = ['Spline']
        list_models = [spline]

    # list_model_names = ['Linear', 'Spline', 'Seasonal']
    # list_models      = [linear, spline, seasonal]

    materials = {k:defaultdict(list) for k in list_model_names}

    nb_chunks = 0

    t_start = time()

    # iterate through the test dataloader:
    for substep, batch in enumerate(test_loader):

        # 1. extract tensors from batch:
        series_c    = batch.get('context_features').to(device)  # [N, T_in, 1] (context values)
        series_t    = batch.get('target_features').to(device)   # [N, T, 1] (target values)
        y_true      = batch.get('missing_features').to(device)  # [N, T_out, 1] (target values)

        coords_c = batch.get('context_coordinates').to(device)  # [N, T_in, 1] (context grid coordinates)
        coords_t = batch.get('target_coordinates').to(device)   # [N, T, 1] (target grid coordinates)
        mask     = batch.get('is_missing_mask').to(device)      # [N, T, 1] (where the samples are not observed)

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

        dict_quantiles = dict()
        for model_name, model in zip(list_model_names, list_models):

            # 3. make predictions:
            if model_name == 'Spline':
                dict_quantiles[model_name] = spline(coords_t.squeeze(), series_t.squeeze(), mask.squeeze())
            else:
                dict_quantiles[model_name] = model(series_t.squeeze(), mask.squeeze()) # type: ignore

        # 4. compute metrics on normalized data:

        y_true = scaler.transform(y_true)[std_mask]

        # update gluonts metrics:
        for model_name, q in dict_quantiles.items():
            dict_evaluators[model_name] = update_gluonts_metrics(
                ytrue           = y_true,
                yhat            = q[mask].reshape(len(mask),-1,1),
                evaluators      = dict_evaluators[model_name]
            )

            if make_plots:
                materials[model_name]['coords_c'].append(coords_c.detach().cpu().squeeze())
                materials[model_name]['series_c'].append(series_c.detach().cpu().squeeze())
                materials[model_name]['coords_t'].append(coords_t.detach().cpu().squeeze())
                materials[model_name]['pred'].append(q.detach().cpu().squeeze())
                materials[model_name]['mask'].append(mask.cpu().squeeze())
                materials[model_name]['series_t'].append(series_t.detach().cpu().squeeze())
        nb_chunks += len(series_t)
        
    # end of tinference loop:
    t_end = time()

    # prepare output dir:
    save_dir_metrics = infer_path / 'metrics'
    save_dir_metrics.mkdir(exist_ok=True)

    # build dict of aggregated metrics and export:
    names = list( dict_evaluators['Linear'].keys() )
    metrics = pd.DataFrame( {
        'Model'  : list(dict_quantiles.keys()),
        'Chunks' : [nb_chunks] * len(dict_quantiles)
    } | {
        name:[dict_evaluators[k][name].get().mean() for k in dict_quantiles.keys()] for name in names
    } | {
        'Time (s)' : [t_end - t_start] * len(dict_quantiles)
    } )
    metrics.set_index('Model', inplace=True)
    metrics.to_csv( save_dir_metrics / 'gluonts_metrics.csv' )
    
    # concat batches outputs
    materials = {mname:{k:torch.cat(val,dim=0) for k,val in mtrl.items() if len(val)>0} for mname ,mtrl in materials.items()}

    

