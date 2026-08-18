from collections import defaultdict

from typing import List, Optional
from pathlib import Path

from time import time
import pandas as pd

from einops import rearrange
import torch
from torch.utils.data import DataLoader

from chronos import Chronos2Pipeline, ChronosBoltPipeline

from src.data.scaler import CustomStandardScaler
from src.tools.utils.plot import plot_forecasts
from src.tools.inference.common import initialize_gluonts_metrics, update_gluonts_metrics


def infer_fn(
    model: Chronos2Pipeline,
    test_loader: DataLoader,
    horizon_len: int,
    path_results_exp: Path,
    bolt_model: Optional[ChronosBoltPipeline] = None,
    quantile_levels: List[float] = [0.1, 0.5, 0.9],
    make_plots: bool = True,
    nb_plots: int = 3,
    max_points_to_plot: int = 672
) -> None:
    
    """
    Main function for running Chronos2 at inference (with or without covariate).
    https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb

    Parameters:
        model (Chronos2Pipeline): pretrained Chronos v2
        test_loader (DataLoader): test dataloader
        horizon_len (int): actual horizon length to predict
        path_results_exp (Path): path where to export results
        bolt_model (ChronosBoltPipeline): pretrained Chronos Bolt
        quantile_levels (List[float]): list of quantile_levels for probabilistic head
        make_plots (bool): make forecasting plots or not
        nb_plots (int): number of samples to plot
        max_points_to_plot (int): restrict plot to the last `max_points_to_plot` timesteps
    """

    infer_path = path_results_exp
    infer_path.mkdir(exist_ok=True)

    materials    = defaultdict(list[torch.Tensor])

    all_models = ['Chronos Bolt', 'Chronos-2', 'Chronos-2 (w. covar.)'] 
    dict_evaluators = {k:initialize_gluonts_metrics(axis=None) for k in all_models}

    nb_chunks = 0

    t_start = time()
    t_cov, t_bolt = 0, 0

    # iterate through the test dataloader:
    for substep, batch in enumerate(test_loader):

        series_c    = batch.get('context_features')             # [N, T_in, 1] (context values)
        series_t    = batch.get('target_features')              # [N, T, 1] (target values)
        y_true      = batch.get('missing_features')             # [N, T_out, 1] (target values)

        # z-normalize context series:
        scaler   = CustomStandardScaler(dim=1,epsilon=1e-7)
        scaler.fit(series_c)
        series_c = scaler.transform(series_c)
        
        # build mask to discard flat segments:
        std_mask =  (scaler.std > 1e-6).squeeze()    # [N,]

        coords_c = batch.get('context_coordinates')  # [N, T_in, 1] (context grid coordinates)
        coords_t = batch.get('target_coordinates')   # [N, T_out, 1] (target grid coordinates)
        mask     = batch.get('is_missing_mask')      # [N, T_out, 1] (where the samples are not observed)

        # make predictions:
        series_c = rearrange( series_c, 'n T 1 -> n 1 T' )[std_mask] # [N', 1, T_in]
        series_t = scaler.transform(series_t)[std_mask] # [N', T_out, 1]

        inputs = series_c
        quantiles, mean = model.predict_quantiles(
            inputs                  = inputs,
            prediction_length       = horizon_len,
            quantile_levels         = quantile_levels,
            context_length          = series_c.shape[-1],
            batch_size              = len(series_c)
            # predict_batches_jointly = False
        )
        interpo = rearrange( torch.stack(mean), 'B 1 T -> B T 1')  # [N', H, 1]
        yhat    = series_t.clone()                     # [N', T_out, 1]
        yhat[:,-horizon_len:,:] = interpo              # [N', T_out, 1]
        dict_quantiles = {'Chronos-2': torch.cat(quantiles, dim=0)} # [N', H, q]

        if bolt_model is not None:
            t0 = time()
            quantiles_bolt, mean_bolt = bolt_model.predict_quantiles(
                inputs                  = series_c.squeeze(), # [N', T_in]
                prediction_length       = horizon_len,
                quantile_levels         = quantile_levels,
            )
            dict_quantiles['Chronos Bolt'] = quantiles_bolt # [N', H, q]
            t_bolt += time() - t0

        # add covariates if available in the batch:
        # `past_covariates` (optional): Dict mapping covariate names to 1D arrays of length `history_length`
        # `future_covariates` (optional): Dict mapping covariate names to 1D arrays of length `prediction_length`
        has_covar = 'covariates' in list(batch.keys())
        if has_covar:
            t0 = time()

            covar = {}
            for covar_name, x in batch['covariates'].items():
                scaler_cov = CustomStandardScaler(dim=1,epsilon=1e-7)
                scaler_cov.fit(x)
                covar[covar_name] = scaler_cov.transform(x)

            inputs = [
                {
                    'target'            : series_c[idx],
                    'past_covariates'   : {
                        covar_name:x[idx,:-horizon_len].squeeze() for covar_name, x in covar.items()
                    },
                    'future_covariates' : {
                        covar_name:x[idx,-horizon_len:].squeeze() for covar_name, x in covar.items()
                    }
                }
                for idx in range(len(series_c))
            ]

            quantiles_covar, mean = model.predict_quantiles(
                inputs                  = inputs,
                prediction_length       = horizon_len,
                quantile_levels         = quantile_levels,
                context_length          = series_c.shape[-1],
                batch_size              = len(series_c),
                # predict_batches_jointly = False
            )
            interpo_covar   = rearrange( torch.stack(mean), 'B 1 T -> B T 1')  # [N', H, 1]
            yhat_cov        = series_t.clone()                 # [N', T_out, 1]
            yhat_cov[:,-horizon_len:,:] = interpo_covar        # [N', T_out, 1]
            dict_quantiles['Chronos-2 (w. covar.)'] = torch.cat(quantiles_covar, dim=0)
            t_cov += time() - t0

        y_true = scaler.transform(y_true)[std_mask]
        mask   = mask[std_mask]
        
        # update gluonts metrics:
        for model_name, q in dict_quantiles.items():
            dict_evaluators[model_name] = update_gluonts_metrics(
                ytrue           = y_true,
                yhat            = q[std_mask],
                evaluators      = dict_evaluators[model_name]
            )
        nb_chunks += len(series_t)
        
        if make_plots:
            materials['coords_c'].append(coords_c[std_mask].detach().cpu().squeeze())
            materials['series_c'].append(series_c.detach().cpu().squeeze())
            materials['coords_t'].append(coords_t[std_mask].detach().cpu().squeeze())
            materials['pred'].append(yhat.detach().cpu().squeeze())
            materials['quantiles'].append(dict_quantiles['Chronos-2'].detach().cpu())
            materials['mask'].append(mask.cpu().squeeze())
            materials['series_t'].append(series_t.detach().cpu().squeeze())
            t0 = time()
            if has_covar:
                materials['pred_cov'].append(yhat_cov.detach().cpu().squeeze())
                materials['quantiles_cov'].append(dict_quantiles['Chronos-2 (w. covar.)'].detach().cpu())
                # scaler_cov = CustomStandardScaler(dim=1,epsilon=1e-7)
                # covar  = list(batch['covariates'].values())[0]
                # scaler_cov.fit(covar)
                # covar = scaler_cov.transform(covar)
                materials['covariates'].append(list(covar.values())[0].detach().cpu().squeeze())
            t_cov += time() - t0

        # # undo z-normalization (back to data space):
        # series_t    = scaler.inv_transform(series_t, std_mask) # [N', T_out, 1]
        # yhat        = scaler.inv_transform(yhat, std_mask)     # [N', T_out, 1]
        
    # end of tinference loop:
    t_end = time()

    # prepare output dir:
    save_dir_metrics = infer_path / 'metrics'
    save_dir_metrics.mkdir(exist_ok=True)

    # build dict of aggregated metrics and export:
    names = list( dict_evaluators['Chronos-2'].keys() )
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
    materials = {key:torch.cat(val,dim=0) for key,val in materials.items() if len(val)>0}

    if make_plots:
        print('plot forecast chronos to check')
        plot_forecasts(
            materials  = materials,
            nb_plots   = nb_plots,
            save_dir   = infer_path,
            gt_exists  = True,
            max_points = max_points_to_plot,
            model_name = 'Chronos-2',
            show_context_points = False,
            plot_covar          = has_covar,
            plot_iqr            = False,
            quantile_levels     = quantile_levels
        )

    return
