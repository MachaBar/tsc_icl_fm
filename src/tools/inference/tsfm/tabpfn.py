import logging
from collections import defaultdict

from typing import Union, List, Literal
from pathlib import Path

from time import time
import pandas as pd

import torch
from torch.utils.data import DataLoader

from tabpfn import TabPFNRegressor

from src.data.scaler import CustomStandardScaler
from src.modules.tabpfn import FourierTokenizer
from src.tools.utils.plot import plot_forecasts, plot_imputations
from src.tools.inference.common import initialize_gluonts_metrics, update_gluonts_metrics

def infer_fn(
    ckpt_path: Path,
    test_loader: DataLoader,
    path_results_exp: Path,
    max_grid_len_in_days: int,
    nb_timesteps_per_day: int,
    list_freq_in_days: List[Union[float, int]] = [1, 7],
    make_plots: bool = True,
    nb_plots: int = 3,
    max_points_to_plot: int = 335,
    infer_setting_name: Literal['forecasting', 'blockwise', 'pointwise'] = 'pointwise'
) -> None:
    """
    Main function for running TabPFN-TS at inference (with or without covariate).

    Args:
        ckpt_path (Path): Path to TabPFNRegressor ckpt
        test_loader (DataLoader): test dataloader
        path_results_exp (Path): path where to export results
        max_grid_len_in_days (int): total grid length, in days
        nb_timesteps_per_day (int): number of timesteps in one day
        list_freq_in_days (List[float | int]): seasonalities of the Fourier Features
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
    logger.info(f'[Device] inference performs on {device}.')

    # instantiate tokenizer:
    tokenizer = FourierTokenizer(
        max_grid_len_in_days = max_grid_len_in_days,
        nb_timesteps_per_day = nb_timesteps_per_day,
        list_freq_in_days    = list_freq_in_days
    )

    # instantiate regressor:
    regressor = TabPFNRegressor(
        model_path = ckpt_path,
        device = device,
        ignore_pretraining_limits = False
    )

    version    = 'v2' if '-v2-' in str(regressor.model_path) else 'v2.5' if '-v2.5-' in str(regressor.model_path) else '?'
    model_name = 'TabPFN-{}'.format(version) 

    logger.info('[Inference] Successfully loaded {} pretrained regressor.'.format(model_name))

    materials       = defaultdict(list[torch.Tensor])
    all_models      = [model_name, model_name + ' (w. covar.)'] 
    dict_evaluators = {k:initialize_gluonts_metrics(axis=None) for k in all_models}

    nb_chunks = 0

    t_start = time()
    t_cov = 0

    # iterate through the test dataloader:
    for substep, batch in enumerate(test_loader):

        # 1. extract tensors from batch:
        series_c = batch.get('context_features').to(device)     # [N, T_in, 1] (context values)
        series_t = batch.get('target_features').to(device)      # [N, T, 1] (target values)
        coords_c = batch.get('context_coordinates').to(device)  # [N, T_in, 1] (context grid coordinates)
        # coords_t = batch.get('target_coordinates').to(device)   # [N, T, 1] (target grid coordinates)
        mask     = batch.get('is_missing_mask').to(device)      # [N, T, 1] (where the samples are not observed)

        coords_t = batch.get('missing_coordinates').to(device)   # [N, T_out, 1] (target grid coordinates)
        y_true = batch.get('missing_features').to(device)       # [N, T_out, 1] (target values)

        # 2. prepare data (normalize & filter):

        # z-normalize context series:
        scaler   = CustomStandardScaler(dim=1,epsilon=1e-7)
        scaler.fit(series_c)

        # get mask for flat windows:
        std_mask = (scaler.std > 1e-6).squeeze()    # [N,]
        series_c = scaler.transform(series_c)[std_mask]

        # z-normalize target series:
        series_t = scaler.transform(series_t)[std_mask] # [N', T, 1]
        y_true   = scaler.transform(y_true)[std_mask]   # [N', T_out, 1]

        # filter out flat contexts:
        coords_c, coords_t, mask = coords_c[std_mask], coords_t[std_mask], mask[std_mask]

        nb_chunks += len(series_t)

        # 3. get features for the target variable:

        # get Fourier features (encoding of time):
        out_tokenizer = tokenizer(coords_c, coords_t)
        H_context     = out_tokenizer['H_context'] # [*, T_in, 1, 5]
        H_target      = out_tokenizer['H_target']  # [*, T_out, 1, 5]

        # 4. prepare covariates, if provided:

        has_covar = 'covariates' in list(batch.keys())
        if has_covar:
            t0 = time()
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

            # # get context vs target coordinates: XXX a revoir en fonction dataloader covar
            # idx_c        = torch.nonzero( coords_c[..., None] == coords_t[0,:,0] )
            # cov_mask_c   = idx_c[:,-1].reshape((len(coords_c),-1,1))
            # covariates   = scaler_cov.transform(covariates) # [*, T_out, C] 
            # covariates_c = covariates.gather(1, cov_mask_c) # [*, T_in, C]
            t_cov += time() - t0
        
        # 5. make predictions:

        # loop through every sample in the batch:
        yhat, yhat_cov = [], []
        for sample in range(len(series_t)):

            # print(f'sample number is {sample}')

            x_train = H_context[sample].squeeze()   # [T_in, 5] # pyright: ignore[reportOptionalSubscript] 
            y_train = series_c[sample].squeeze()    # [T_in,]
            x_query = H_target[sample].squeeze()    # [T_out, 5] # pyright: ignore[reportOptionalSubscript] 

            # (a) univariate setting:
             
            # fit context:
            regressor.fit(x_train.cpu().detach(), y_train.cpu().detach())

            # predict:
            forecast = regressor.predict(
                x_query.cpu().detach(),
                output_type = 'quantiles',
                quantiles   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            )
            yhat.append( torch.stack([torch.Tensor(x) for x in forecast], dim=-1).to(device) )

            # (b) with covariate:
            if not has_covar:
                continue
            
            t0 = time()
            x_train_cov = torch.cat([covariates[sample,~mask[sample]].unsqueeze(-1), x_train], dim=1) # [T_in, 5+C]
            x_query_cov = torch.cat([covariates[sample, mask[sample]].unsqueeze(-1), x_query], dim=1) # [T_out, 5+C]
            # x_train_cov = torch.cat([covariates_c[sample], x_train], dim=1) # [T_in, 5+C]
            # x_query_cov = torch.cat([covariates[sample], x_query], dim=1)   # [T_out, 5+C]

            # fit context:
            regressor.fit(x_train_cov.cpu().detach(), y_train.cpu().detach())

            # predict:
            forecast = regressor.predict(
                x_query_cov.cpu().detach(),
                output_type = 'quantiles',
                quantiles   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            )
            yhat_cov.append( torch.stack([torch.Tensor(x) for x in forecast], dim=-1).to(device) )
            t_cov += time() - t0

        # stack all predictions:
        dict_quantiles = {model_name: torch.stack(yhat, dim=0)}
        if has_covar:
            dict_quantiles[all_models[-1]] = torch.stack(yhat_cov, dim=0)
        
        # 6. compute metrics on normalized data:
        # results     = update_results(series_t, yhat, mask, results, is_normalized=True)
        # results_cov = update_results(series_t, yhat_cov, mask, results_cov, is_normalized=True)

        # update gluonts metrics:
        for name, quantiles in dict_quantiles.items():
            dict_evaluators[name] = update_gluonts_metrics(
                ytrue           = y_true,
                yhat            = quantiles[std_mask],
                evaluators      = dict_evaluators[name]
            )

        if make_plots:
            materials['coords_c'].append(coords_c[std_mask].detach().cpu().squeeze())
            materials['series_c'].append(series_c.detach().cpu().squeeze())
            materials['coords_t'].append(coords_t[std_mask].detach().cpu().squeeze())
            materials['pred'].append(dict_quantiles[model_name].detach().cpu().squeeze())
            materials['mask'].append(mask.cpu().squeeze())
            materials['series_t'].append(series_t.detach().cpu().squeeze())
            t0 = time()
            if has_covar:
                materials['pred_cov'].append(dict_quantiles[all_models[-1]].detach().cpu().squeeze())
                scaler_plot = CustomStandardScaler(dim=1,epsilon=1e-7)
                covar  = list(batch['covariates'].values())[0]
                scaler_plot.fit(covar)
                covar = scaler_plot.transform(covar)
                materials['covariates'].append(covar.detach().cpu().squeeze())
            t_cov += time() - t0

        # 7. compute metrics on raw data:

        # # undo z-normalization (back to data space) and compute metrics:
        # series_t = scaler.inv_transform(series_t, std_mask) # [N', T_out, 1]        
        # yhat     = scaler.inv_transform(yhat.to(device), std_mask)     # [N', T_out, 1]
        # results  = update_results(series_t, yhat, mask, results, is_normalized = False)

    # end of tinference loop:
    t_end = time()

    save_dir_metrics = infer_path / 'metrics'
    save_dir_metrics.mkdir(exist_ok=True)

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
            'plot_iqr'        : False,
            'quantile_levels' : [0.1, 0.5, 0.9]
        }
        plot_fn(
            materials           = materials,
            nb_plots            = nb_plots,
            save_dir            = infer_path,
            gt_exists           = True,
            show_context_points = True,
            max_points          = max_points_to_plot,
            model_name          = model_name,
            plot_covar          = has_covar,
            **kwargs
        )

    return