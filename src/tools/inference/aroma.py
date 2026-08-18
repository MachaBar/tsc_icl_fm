from collections import defaultdict

from typing import Literal, List
from pathlib import Path
from time import time

from einops import rearrange, repeat

import pandas as pd

import torch
from torch.utils.data import DataLoader

from src.modules.aroma import AROMAEncoderDecoderKL, AROMAEncoderDecoderICL
from src.data.scaler import CustomStandardScaler
from src.tools.utils.plot import plot_imputations, plot_forecasts
from src.tools.inference.common import initialize_gluonts_metrics, update_gluonts_metrics

def infer_fn(
    inr: AROMAEncoderDecoderKL | AROMAEncoderDecoderICL,
    test_loader: DataLoader,
    path_results_exp: Path,
    make_plots: bool = True,
    nb_plots: int = 3,
    export_outputs: bool = False,
    max_points_to_plot: int = 672,
    infer_setting_name: Literal['forecasting', 'blockwise', 'pointwise'] = 'pointwise',
    quantile_median: int = 0,
    quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    validation_mode: bool = False
):
    
    """
    Main function for running an INR model at inference.

    Parameters:
        inr (AROMAEncoderDecoderKL): INR network to evaluate
        test_loader (DataLoader): test dataloader
        path_results_exp (Path): path where to export results
        make_plots (bool): whether to export inference plots
        nb_plots (int): number of samples to plot
        export_outputs (bool): if True, export reconstruction, grids etc.
        max_points_to_plot (int): length of the samples to plots
        infer_setting_name (Literal['forecasting', 'blockwise', 'pointwise']): name of the inference setting

    Returns:
        results (dict): dict of outputs with following items:
            - mae_on_missing_values (float): MAE on all target timesteps with NaN (if ground truth provided)
            - mae_on_observed_values (float): MAE on all target timesteps with no NaN
            - reco (torch.Tensor): estimated features on the target grid
            - gt (torch.Tensor): ground truth features on the target grid, if provided
    """

    infer_path = path_results_exp
    infer_path.mkdir(exist_ok=True)

    # get device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # move model to device:
    inr = inr.to(device)

    results   = defaultdict(list[torch.Tensor])
    materials = defaultdict(list[torch.Tensor])

    model_name = 'TS_ICL'
    all_models      = [model_name, model_name + ' (w. covar.)'] 
    dict_evaluators = {k:initialize_gluonts_metrics(axis=None) for k in all_models}


    nb_chunks = 0
    t_start   = time()

    # iterate through the test dataloader:
    with torch.no_grad():
        
        for substep, batch in enumerate(test_loader):
            
            inr.eval()

            # 1/6 extract tensors from batch:
            series_c = batch.get('context_features').to(device)    # [N, T_in, 1] (context values)
            series_t = batch.get('target_features')                # [N, T, 1] (target values)

            # check if series_t (ground truth) exists:
            series_t_exists = isinstance(series_t, torch.Tensor)
            assert series_t_exists

            # 2/6 prepare data (normalize):

            # z-normalize context series:
            scaler   = CustomStandardScaler(dim=1,epsilon=1e-5)
            scaler.fit(series_c)
            series_c = scaler.transform(series_c)

            # build mask to discard flat segments:
            std_mask =  (scaler.std > 1e-5).reshape(-1,) #.squeeze()

            # load the rest of the batch:
            coords_c    = batch.get('context_coordinates').to(device)  # [N, T_in, 1] (context grid coordinates)
            coords_t    = batch.get('target_coordinates').to(device)   # [N, T, 1] (target grid coordinates)
            mask        = batch.get('is_missing_mask').to(device)      # [N, T, 1] (where the samples are not observed)

            # 3/6 look for covariates
            has_covar = 'covariates' in list(batch.keys())
            if has_covar:
                # fetch all covariates:
                covariates = torch.cat(
                    [
                        x.to(device) for x in batch['covariates'].values() # [*, T, 1]
                    ],
                    dim = -1
                ) # [*, T_out, C]

                # z-normalize covariate series:
                scaler_cov = CustomStandardScaler(dim=1,epsilon=1e-7)
                scaler_cov.fit(covariates)
                covariates   = scaler_cov.transform(covariates) # [*, T, C]
                covariates   = rearrange(covariates, 'b t c -> b c t 1')

                # coords_cov = repeat(coords_t, 'b t 1 -> b c t 1', c = covariates.shape[1])

            # 4/6 make predictions

            # build query coords:
            query_coords = torch.cat([coords_c, coords_t], dim=1) # [N, T_in+T, 1]

            # get predictions:
            outputs, _ = inr(
                series               = series_c,
                coords               = coords_c,
                target_coords        = query_coords,
                undo_asinh_transform = True
            )

            if has_covar:
                covariates_c = covariates[~mask.unsqueeze(1)].reshape(len(covariates),covariates.shape[1],-1,1)
                # covariates = torch.cat([covariates_c, covariates], dim=2)
                outputs_cov, _ = inr(
                    series               = series_c,
                    coords               = coords_c,
                    target_coords        = query_coords,
                    series_covar         = torch.cat([covariates_c, covariates], dim=2),
                    coords_covar         = query_coords, # query_coords,
                    # coords_covar     = coords_cov,
                    undo_asinh_transform = True

                )
                    
            if inr.apply_asinh_transform:
                series_c  = torch.sinh(series_c)

            dict_quantiles = {model_name: outputs}
            if has_covar:
                dict_quantiles[all_models[-1]] = outputs_cov

            # 5/6 compute metrics on normalized data

            interpo = outputs[..., quantile_median:quantile_median+1]
            # interpo = torch.mean(outputs[:,n_obs:], dim=2, keepdim=True)

            # Compute normalize score 
            series_t        = scaler.transform(series_t.to(device))
            error_normalize = torch.abs(interpo - series_t)[std_mask]
            mask            = mask[std_mask]

            # XXX soon deprecated, replaced by gluonts metrics (on missing values)
            mae_on_missing_values_norm  = error_normalize[mask]
            mae_on_observed_values_norm = error_normalize[~mask]
            mse_on_missing_values_norm  = error_normalize[mask]**2
            mse_on_observed_values_norm = error_normalize[~mask]**2

            # update gluonts metrics:
            for name, quantiles in dict_quantiles.items():
                dict_evaluators[name] = update_gluonts_metrics(
                    ytrue           = series_t[std_mask],
                    yhat            = quantiles[std_mask],
                    evaluators      = dict_evaluators[name],
                    is_target_mask  = mask
                )

            nb_chunks += len(error_normalize)

            if make_plots:
                materials['coords_c'].append(coords_c[std_mask].detach().cpu().squeeze())
                materials['series_c'].append(series_c[std_mask].detach().cpu().squeeze())
                materials['coords_t'].append(coords_t[std_mask].detach().cpu().squeeze())
                materials['pred'].append(interpo[std_mask].detach().cpu().squeeze())
                materials['quantiles'].append(outputs[std_mask].detach().cpu().squeeze())
                materials['mask'].append(mask.cpu().squeeze())
                materials['series_t'].append(series_t[std_mask].detach().cpu().squeeze())
                if has_covar:
                    materials['pred_cov'].append(outputs_cov[..., quantile_median:quantile_median+1][std_mask].detach().cpu().squeeze())
                    materials['quantiles_cov'].append(outputs_cov[std_mask].detach().cpu().squeeze())
                    materials['covariates'].append(covariates[std_mask].detach().cpu().squeeze())

            # 6/6 compute metrics on raw data

            # undo z-normalization (back to data space):
            interpo = scaler.inv_transform(interpo)

            # compute errors on missing values (if ground truth is provided):
            series_t = scaler.inv_transform(series_t)          # [N, T_in, 1]
            error    = torch.abs(interpo - series_t)[std_mask] # MAE

            mae_on_missing_values  = error[mask]        # Forecasting -> MAE on horizon
            mae_on_observed_values = error[~mask]       # Forecasting -> MAE on lookback
            mse_on_missing_values  = error[mask]**2     # Forecasting -> MSE on horizon
            mse_on_observed_values = error[~mask]**2    # Forecasting -> MSE on lookback

            # XXX soon deprecated, replaced by gluonts metrics (on missing values)
            
            # denormalized errors:
            if len(mae_on_missing_values) > 0:
                results['mae_on_missing_values'].append(mae_on_missing_values)
                results['mse_on_missing_values'].append(mse_on_missing_values)
            if len(mae_on_observed_values) >0:
                results['mae_on_observed_values'].append(mae_on_observed_values)
                results['mse_on_observed_values'].append(mse_on_observed_values)

            # normalized errors:
            if len(mae_on_missing_values_norm) > 0:
                results['mae_on_missing_values_norm'].append(mae_on_missing_values_norm)
                results['mse_on_missing_values_norm'].append(mse_on_missing_values_norm)
            if len(mae_on_observed_values_norm) >0:
                results['mae_on_observed_values_norm'].append(mae_on_observed_values_norm)
                results['mse_on_observed_values_norm'].append(mse_on_observed_values_norm)

    # agregate all results:
    results = {key:torch.cat(val,dim=0).cpu().detach() for key,val in results.items() if len(val)>0}
    results = {key:val.nanmean().cpu().detach().item() if  ('mae' in key or 'mse' in key) and len(val)>0 else val for key,val in results.items()}
    
    t_end = time()

    if validation_mode:
        return results.get('mae_on_missing_values_norm') # type: ignore
    
    save_dir_metrics = infer_path / 'metrics'
    save_dir_metrics.mkdir(exist_ok=True)
    
    # export metrics:
    df = pd.DataFrame({
        'Model': ['AROMA'],
        'Chunks': [nb_chunks],
        'MAE on Missing Values' : [results.get('mae_on_missing_values')],
        'MAE on Observed Values': [results.get('mae_on_observed_values')],
        'MAE on Missing Values (norm)' : [results.get('mae_on_missing_values_norm')],
        'MAE on Observed Values (norm)': [results.get('mae_on_observed_values_norm')],
        'Time (s)': [t_end-t_start]
    })
    df.set_index('Model', inplace=True)
    df.to_csv(save_dir_metrics / 'mae.csv')

    df = pd.DataFrame({
        'Model': ['AROMA'],
        'Chunks': [nb_chunks],
        'MSE on Missing Values' : [results.get('mse_on_missing_values')],
        'MSE on Observed Values': [results.get('mse_on_observed_values')],
        'MSE on Missing Values (norm)' : [results.get('mse_on_missing_values_norm')],
        'MSE on Observed Values (norm)': [results.get('mse_on_observed_values_norm')],
        'Time (s)': [t_end-t_start]
    })
    df.set_index('Model', inplace=True)
    df.to_csv(save_dir_metrics / 'mse.csv')
    
    # build dict of aggregated metrics and export:
    names = list( dict_evaluators[model_name].keys() )
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
        plot_fn = plot_forecasts if infer_setting_name == 'forecasting' else plot_imputations
        kwargs  = {
            'is_blockwise'    : infer_setting_name == 'blockwise',
            'plot_iqr'        : outputs.shape[-1] > 1,
            'quantile_levels' : quantile_levels
        }
        plot_fn(
            materials           = materials,
            nb_plots            = nb_plots,
            save_dir            = infer_path,
            gt_exists           = series_t_exists,
            show_context_points = False if infer_setting_name == 'forecasting' else True,
            max_points          = max_points_to_plot,
            model_name          = 'TS-ICL',
            plot_covar          = has_covar,
            **kwargs
        )

    if export_outputs:
        save_dir_materials = infer_path / 'materials'
        save_dir_materials.mkdir(exist_ok=True)
        torch.save(materials, save_dir_materials / 'materials.pt')

    return

