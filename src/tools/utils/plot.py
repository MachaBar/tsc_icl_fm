from typing import List, Dict, Optional 
from pathlib import Path

import torch 
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

def plot_losses(
    l_epochs: List[int],
    dict_losses: Dict[str, List[float]],
    filename: Optional[str] = 'training_loss',
    save_path: Optional[Path] = Path('./'),
    clip_yaxis: bool = False
) -> None:

    fig, ax = plt.subplots(1,1,figsize=(12,3)) # (20,5)
    ax.yaxis.grid(True)
    for key, l_loss in dict_losses.items():
        ax.semilogy(l_epochs, l_loss, label=key)
    if clip_yaxis:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim((ymin, min(ymax, 1.1)))
    ax.legend()
    fig.tight_layout()
    fig.savefig( save_path / '{}.png'.format(filename) )
    plt.close(fig)


def plot_loss_steps(
    step_loss: List[float],
    filename: Optional[str] = 'training_loss_step',
    save_path: Optional[Path] = Path('./'),
    clip_yaxis: bool = False
) -> None:
    
    fig,ax = plt.subplots(1,1,figsize=(12,3),sharey=True)
    ax.yaxis.grid(True)

    # step_loss = np.asarray(step_loss)

    ax.loglog(step_loss, label='values',color='tab:gray', lw=0.75)
    try:
        M = 16
        running_average = np.convolve(step_loss, np.ones(M)/M, mode='valid')        
        ax.fill_between(np.arange(len(step_loss))[M-1:], 0, running_average, color='tab:blue', alpha=0.1)
        ax.loglog(np.arange(len(step_loss))[M-1:], running_average, color='tab:blue', lw=1,alpha=0.85)
    except:
        pass
    if clip_yaxis:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim((ymin, min(ymax, 1.1)))    
    ax.set_xlabel('Steps')
    ax.legend()
    fig.tight_layout()
    fig.savefig( save_path / '{}.png'.format(filename) )
    plt.close(fig)


def plot_forecasts(
    materials: Dict[str, torch.Tensor],
    nb_plots: int,
    save_dir: Path,
    gt_exists: bool,
    max_points: int = -1,
    show_context_points: bool = False,
    model_name: str = 'AROMA',
    plot_covar: bool = False,
    **kwargs
):
    """
    Generate and save time series forecasting plots.

    Args:
        materials (Dict[str, torch.Tensor]): Data for plotting:
            - 'coords_c', 'series_c': Context points (known values).
            - 'coords_t', 'series_t': Full time series (including ground truth).
            - 'pred': forecasted values by TimeFlow.
            - 'mask': Boolean mask (True for forecasted values).
        nb_plots (int): Number of samples to plot.
        save_dir (Path): Directory for saving plots.
        gt_exists (bool): Whether ground truth is available.
        max_points (int): how many time points to plot (will plot all if <0)

    Output:
        Saves `nb_plots` as PNGs in `save_dir/inference_plots/`.
    """    

    save_dir_plots = save_dir / 'inference_plots'
    save_dir_plots.mkdir(exist_ok=True)

    samples_to_plot = torch.randint(materials['pred'].shape[0], (nb_plots,))
    max_points      = min(max_points, materials['pred'].shape[1]) if max_points > 0 else materials['pred'].shape[1]

    plot_iqr        = kwargs.get('plot_iqr', False)
    quantile_levels = kwargs.get('quantile_levels', [0.1, 0.5, 0.9])

    # set matplotlib params:
    matplotlib.rc('font', **{'size':12})
    matplotlib.rc('lines', **{'linewidth':2.5, 'linestyle': '-', 'markersize': 5})

    for s in samples_to_plot:
        
        # get target coordinates (ordered from 0 to 1 with no repeat):
        coords_t    = np.array(materials['coords_t'][s])

        # limit target to max_points samples for clarity:
        time_mask_t     = coords_t >= coords_t[-max_points]
        coords_t        = coords_t[time_mask_t]
        forecast_values = np.array(materials['pred'][s])[time_mask_t]

        # get target mask (True if value is missing, False if observed and part of context):
        mask = np.array(materials['mask'][s])[time_mask_t]

        # make TimeFlow interpo plot:
        coords_forecast = coords_t[mask]
        values_forecast = forecast_values[mask]

        # init figure:
        fig, ax = plt.subplots(1, 1, figsize=(12,4))

        if plot_iqr:
            fig1, ax1 = plt.subplots(1, 1, figsize=(12,4))

        # plot Ground Truth if accessible:
        if gt_exists:
            ax.plot(
                coords_t,
                materials['series_t'][s][time_mask_t],
                color="tab:green",
                lw=1.75,
                label='Ground Truth'
            )
            if plot_iqr:
                ax1.plot(
                    coords_t[-1008-len(coords_forecast):],
                    materials['series_t'][s][time_mask_t][-1008-len(coords_forecast):],
                    color="tab:green",
                    lw=1.75,
                    label='Ground Truth'
                )


        # plot context points (context = not is missing in target):
        if show_context_points:
            coords_context = coords_t[~mask]
            values_context = materials['series_t'][s][time_mask_t][~mask]

            ax.plot(
                coords_context,
                values_context,
                'o',
                markersize=1.5,
                color="tab:red",
                label='Context'
            )

        if 'covariates' in materials:
            ax.plot(
                coords_t,
                materials['covariates'][s][time_mask_t],
                lw=0.8,
                color="tab:gray",
                alpha=0.5,
                label="Covariate",
            )

        if len(coords_forecast) > 0:

            # render forecast region as light gray:
            ax.axvline(coords_forecast[0], ls='--', color='k', lw=0.75, alpha=0.8)
            ax.axvspan(coords_forecast[0], ax.get_xlim()[1], facecolor='tab:gray', alpha=0.1)

            if plot_iqr:
                ax1.axvline(coords_forecast[0], ls='--', color='k', lw=0.75, alpha=0.8)
                ax1.axvspan(coords_forecast[0], coords_forecast[-1]+0.01, facecolor='tab:gray', alpha=0.1)

            # plot forecast as a line:
            ax.plot(
                coords_forecast,
                values_forecast,
                color = 'tab:blue',
                lw    = 1.75,
                label = model_name + ('(no covar)' if plot_covar else '')
            )
            if plot_covar:
                ax.plot(
                    coords_forecast,
                    np.array(materials['pred_cov'][s])[time_mask_t][mask],
                    color = 'tab:orange',
                    lw    = 1.75,
                    label = model_name + '(with covar)'
                )
            if plot_iqr:
                key = 'quantiles_cov' if plot_covar else 'quantiles'
                all_quantiles = np.array(materials[key][s])[mask]
                ax1.fill_between(
                    coords_forecast,
                    y1    = all_quantiles[...,quantile_levels.index(0.05)],
                    y2    = all_quantiles[...,quantile_levels.index(0.95)],
                    color = 'tab:blue',
                    label = model_name + '(IQR 5-95)',
                    lw    = 0,
                    alpha = 0.2
                )
                ax1.fill_between(
                    coords_forecast,
                    y1    = all_quantiles[...,quantile_levels.index(0.25)],
                    y2    = all_quantiles[...,quantile_levels.index(0.75)],
                    color = 'tab:blue',
                    label = model_name + '(IQR 25-75)',
                    lw    = 0,
                    alpha = 0.5
                )
                ax1.plot(
                    coords_forecast,
                    all_quantiles[...,quantile_levels.index(0.5)],
                    color = 'tab:blue',
                    lw    = 1.75,
                    label = model_name + '(median)'
                )

        ax.legend(
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.075),
            fancybox=True, 
            shadow=True, 
            ncol=3+int(plot_covar)+int('covariates' in materials)
        )
    
        fig.tight_layout()
        fig.savefig(
            f'{save_dir_plots}/sample_{s}.pdf',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close(fig)
        if plot_iqr:
            ax1.legend(
                loc='upper center', 
                bbox_to_anchor=(0.5, -0.075),
                fancybox=True, 
                shadow=True, 
                ncol=4
            )
            fig1.tight_layout()
            fig1.savefig(
                f'{save_dir_plots}/sample_{s}_iqr.pdf',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close(fig1)
    

def plot_imputations(
    materials: Dict[str, torch.Tensor],
    nb_plots: int,
    save_dir: Path,
    gt_exists: bool,
    max_points: int = -1,
    show_context_points: bool = False,
    model_name: str = 'TimeFlow',
    plot_covar: bool = False,
    **kwargs
):

    """
    Generates and saves time series imputation plots.

    Args:
        materials (Dict[str, torch.Tensor]): Data for plotting:
            - 'coords_c', 'series_c': Context points (known values).
            - 'coords_t', 'series_t': Full time series (including ground truth).
            - 'pred': imputed values by TimeFlow.
            - 'mask': Boolean mask (True for forecasted values).
        nb_plots (int): Number of samples to plot.
        save_dir (Path): Directory for saving plots.
        gt_exists (bool): Whether ground truth is available.
        max_points (int): how many time points to plot (will plot all if <0)

    Output:
        Saves `nb_plots` as PNGs in `save_dir/inference_plots/`.
    """    

    save_dir_plots = save_dir / 'inference_plots'
    save_dir_plots.mkdir(exist_ok=True)

    samples_to_plot = torch.randint(materials['pred'].shape[0], (nb_plots,))
    max_points      = min(max_points, materials['pred'].shape[1]) if max_points > 0 else materials['pred'].shape[1]

    is_blockwise = kwargs.get('is_blockwise', False)

    plot_iqr        = kwargs.get('plot_iqr', False)
    quantile_levels = kwargs.get('quantile_levels', [0.1, 0.5, 0.9])


    # set matplotlib params:
    matplotlib.rc('font', **{'size':12})
    matplotlib.rc('lines', **{'linewidth':2.5, 'linestyle': '-', 'markersize': 5})

    for s in samples_to_plot:
        
        # get target coordinates (ordered from 0 to 1 with no repeat):
        coords_t    = np.array(materials['coords_t'][s])

        # limit target to max_points samples for clarity:
        time_mask_t     = coords_t >= coords_t[-max_points]
        coords_t        = coords_t[time_mask_t]
        forecast_values = np.array(materials['pred'][s])[time_mask_t]

        # get target mask (True if value is missing, False if observed and part of context):
        mask = np.array(materials['mask'][s])[time_mask_t]

        # get start and end indices of NA blocks:
        block_indices = np.where(np.diff(mask))[0] if is_blockwise else []

        # init figure:
        fig, ax = plt.subplots(1, 1, figsize=(12,4))

        # plot Ground Truth if accessible:
        if gt_exists:
            ax.plot(
                coords_t,
                materials['series_t'][s][time_mask_t],
                color="tab:green",
                lw=1.75,
                label='Ground Truth'
            )

        if 'covariates' in materials:
            ax.plot(
                coords_t,
                materials['covariates'][s][time_mask_t],
                lw=0.8,
                color="tab:gray",
                alpha=0.5,
                label="Covariate",
            )
        
        # make TimeFlow interpo plot:
        coords_forecast = coords_t[mask]

        if len(coords_forecast) > 0:
            
            for i,idx in enumerate(block_indices[::2]):
                plt.axvline(coords_t[idx],ls='--',color='k',lw=.5)
                end_point = coords_t[block_indices[1::2][i]] if i < len(block_indices[1::2]) else coords_t[-1]
                plt.axvline(end_point,ls='--',color='k',lw=0.5)
                plt.axvspan(coords_t[idx], end_point, facecolor='tab:gray',alpha=0.1)

            # plot TimeFlow at all timesteps (as a line):
            ax.plot(
                coords_t,
                forecast_values,
                color = "tab:blue",
                lw    = 1.5,
                label = model_name  + ('(no covar)' if plot_covar else '')
            )

            if plot_iqr:
                key = 'quantiles_cov' if plot_covar else 'quantiles'
                all_quantiles = np.array(materials[key][s])[time_mask_t]
                ax.fill_between(
                    coords_t,
                    y1    = all_quantiles[...,quantile_levels.index(0.05)],
                    y2    = all_quantiles[...,quantile_levels.index(0.95)],
                    color = 'tab:blue',
                    label = model_name + '(IQR 5-95)',
                    lw    = 0,
                    alpha = 0.3
                )
                ax.fill_between(
                    coords_t,
                    y1    = all_quantiles[...,quantile_levels.index(0.25)],
                    y2    = all_quantiles[...,quantile_levels.index(0.75)],
                    color = 'tab:blue',
                    label = model_name + '(IQR 25-75)',
                    lw    = 0,
                    alpha = 0.7
                )

            # plot forecast as a line:
            if plot_covar:
                ax.plot(
                    coords_t,
                    np.array(materials['pred_cov'][s])[time_mask_t],
                    color = 'tab:orange',
                    lw    = 1.5,
                    label = model_name + '(with covar)'
                )

        # plot context points (context = not is missing in target):
        if show_context_points:
            coords_context = coords_t[~mask]
            values_context = materials['series_t'][s][time_mask_t][~mask]

            ax.plot(
                coords_context,
                values_context,
                'o',
                markersize=2.,
                color="tab:red",
                label='Context'
            )
        
        ax.legend(
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.075),
            fancybox=True, 
            shadow=True, 
            ncol=3+2*int(plot_iqr) + int(plot_covar)#+int('covariates' in materials)
        )
    
        fig.tight_layout()
        fig.savefig(
            f'{save_dir_plots}/sample_{s}.pdf',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close(fig)