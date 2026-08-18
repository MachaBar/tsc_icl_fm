import logging

from collections import defaultdict

from omegaconf import DictConfig
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.scaler import CustomStandardScaler
from src.modules.aroma import AROMAEncoderDecoderKL
from src.modules.icl_learning import ICLearning
from src.optim.losses import MultiQuantileSmoothPinballLoss
from src.optim.scheduler import get_scheduler

from src.tools.utils.mask import std_mask
from src.tools.utils.plot import plot_losses, plot_loss_steps


def train_fn(
    inr: AROMAEncoderDecoderKL,
    tf_icl: ICLearning,
    train_loader: DataLoader,
    cfg_tf_icl: DictConfig,
    cfg_aroma: DictConfig,
    path_results_exp: Path
) -> None:
    """
    Main function for training Transformer ICL model.

    Args:
        inr (AROMAEncoderDecoderKL): freeze INR network 
        tf_icl (ICLearning): Transformer in-context learning network to train
        train_loader (DataLoader): train dataloader
        cfg (DictConfig): yaml config (meta file with all dependencies)
        path_results_exp (Path): path where to export losses & checkpoints
    """

    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get config:
    cfg_optim = cfg_tf_icl.optim
    cfg_train = cfg_tf_icl.trainer

    # Build directories to export losses and checkpoints:
    save_loss_path = path_results_exp / 'loss'
    save_ckpt_path = path_results_exp / 'ckpt'
    (save_loss_path / 'steps').mkdir(exist_ok=True,parents=True)
    (save_loss_path / 'epochs').mkdir(exist_ok=True,parents=True)
    save_ckpt_path.mkdir(exist_ok=True)

    # Move model to device:
    inr    = inr.to(device)
    tf_icl = tf_icl.to(device)

    # define loss function: 
    # loss_type = cfg_optim.get('loss_type','mse').lower()
    # loss_fn   = losses.per_element_huber_fn if loss_type == 'huber' else losses.per_element_mse_fn

    # quantiles loss 
    quantiles = torch.linspace(
        cfg_tf_icl.tf_icl.start_quantile, 
        cfg_tf_icl.tf_icl.end_quantile, 
        cfg_tf_icl.tf_icl.nb_quantiles
        ).to(device)
    
    loss_fn = MultiQuantileSmoothPinballLoss(quantiles=quantiles, alpha=0.01)

    # Define optimizer:
    optimizer = torch.optim.AdamW(
        [
            {"params": tf_icl.parameters(), "lr": cfg_optim.lr},
        ],
        lr=cfg_optim.lr,
        weight_decay=0.01,
    )

    # Define scheduler:
    scheduler = get_scheduler(cfg_optim.scheduler, optimizer)

    # gradient clipping:
    apply_grad_clip = getattr(cfg_optim, 'apply_grad_clip', False)

    # get asinh transform:
    apply_asinh_transform = getattr(cfg_aroma, "apply_asinh_transform", False)

    best_loss = np.inf

    # alloc some space to save metrics:
    per_epoch_callback = defaultdict(list)
    per_step_callback  = defaultdict(list)

    global_steps = 0

    # Iterate through the epochs (one step = one pass of the train loader):
    for step in range(cfg_train.max_epochs):

        # initialize per-epoch metrics:
        fit_loss_epoch   = 0
        mae_target_epoch = 0
        mse_target_epoch = 0

        ntrain = 0
        
        # Iterate through the train dataloader:
        for substep, batch in enumerate(train_loader):

            series_c = batch.get('context_features').to(device, non_blocking=True)     # [N, L, 1] (context values)
            coords_c = batch.get('context_coordinates').to(device, non_blocking=True)  # [N, L, 1] (context grid coordinates)

            series_t    = batch.get('missing_features').to(device, non_blocking=True)    # [N, H, 1] (target values)
            coords_t    = batch.get('missing_coordinates').to(device, non_blocking=True) # [N, H, 1] (target grid coordinates)

            inr.train()
            
            # z-normalize series (fit on context, transform on context + target):
            scaler = CustomStandardScaler(dim=1,epsilon=1e-7)
            scaler.fit(series_c)
            series_c = scaler.transform(series_c)
            series_t = scaler.transform(series_t)

            # if remove weird samples from series:
            values_mask = (series_t.max(1)[0] < 20).squeeze() # change here for removing more XXX
            series_c, series_t = series_c[values_mask], series_t[values_mask]
            coords_c, coords_t = coords_c[values_mask], coords_t[values_mask]
            
            # If apply_mask_during_training, remove samples with very small standard deviation:
            if cfg_optim.apply_mask_during_training:

                _, mask_c = std_mask(series_c, threshold=8.0, return_mask=True)
                _, mask_t = std_mask(series_t, threshold=8.0, return_mask=True)
                mask_std  = torch.logical_and(mask_c, mask_t)
                
                series_c, coords_c, series_t, coords_t = series_c[mask_std], coords_c[mask_std], series_t[mask_std], coords_t[mask_std]
                
            n_samples = series_c.shape[0]
            if n_samples == 0:
                continue

            # optionally, apply asinh transform:
            if apply_asinh_transform:
                series_c = torch.asinh( series_c )
                series_t = torch.asinh( series_t )

            # AROMA forward pass (encoder + decoder):
            with torch.no_grad():
                _, hidden_features_context = inr(
                    images=series_c,
                    coords=coords_c,
                    target_coords=coords_c,
                    return_stats=False,
                    sample_posterior=False,
                    return_act=True
                )
                _, hidden_features_target = inr(
                    images=series_c,
                    coords=coords_c,
                    target_coords=coords_t,
                    return_stats=False,
                    sample_posterior=False,
                    return_act=True
                )
            
            hidden_features_context = hidden_features_context[-1]  # get last INR layer (B, T_observed, d)
            hidden_features_target  = hidden_features_target[-1]   # get last INR layer (B, T_missing, d)

            hidden_features = torch.cat([hidden_features_context, hidden_features_target], dim=1) #(B, T, d)        

            # forward pass through Transformer ICL block
            yhat_target = tf_icl(hidden_features, series_c) # (B, T_target, Q)

            loss = loss_fn(yhat_target, series_t).mean()

            # do gradient descent wrt all parameters:
            optimizer.zero_grad()
            loss.backward()

            # optionally, do gradient clipping:
            if apply_grad_clip:
                nn.utils.clip_grad_value_(inr.parameters(), clip_value=1.0)

            # optmizer step:
            optimizer.step()

            with torch.no_grad():

                # get median prediction (B, T_target, 1)
                yhat_target = yhat_target[..., cfg_tf_icl.tf_icl.quantile_median].unsqueeze(-1) 

                # optionnally, undo asinh transform to get metrics in z-norm space:
                if apply_asinh_transform:
                    series_t, yhat_target   = torch.sinh(series_t), torch.sinh(yhat_target)

                # get MAE/MSE on unobserved coords only:
                loss_target_mae   = torch.nn.L1Loss()(yhat_target, series_t).cpu().detach()
                mae_target_epoch += loss_target_mae.item() * n_samples
                loss_target_mse   = torch.nn.MSELoss()(yhat_target, series_t).cpu().detach()
                mse_target_epoch += loss_target_mse.item() * n_samples

            # store step losses:
            per_step_callback['loss'].append(loss.item())
            per_step_callback['mse_target'].append(loss_target_mse.item())
            per_step_callback['mae_target'].append(loss_target_mae.item())

            # update total loss:
            fit_loss_epoch += loss.item() * n_samples
           
            ntrain       += n_samples
            global_steps += 1

            # plot losses (stepwise):
            if (global_steps % 500 == 0 or global_steps == int(cfg_train.max_epochs * len(train_loader)) - 1):
                plot_loss_steps(
                    per_step_callback['loss'],
                    filename   = 'loss_step_{}'.format(cfg_tf_icl.data.name.lower()),
                    save_path  = save_loss_path / 'steps',
                    clip_yaxis = True
                )
                [
                    plot_loss_steps(
                        per_step_callback['{}_{}'.format(l,segment)],
                        filename   = '{}_{}_step_{}'.format(l,segment,cfg_tf_icl.data.name.lower()),
                        save_path  = save_loss_path / 'steps',
                        clip_yaxis = True
                    ) for l in ['mae', 'mse'] for segment in ['target']
                ]

        global_fit_train_mse = fit_loss_epoch
        global_mae_target    =  mae_target_epoch
        global_mse_target    = mse_target_epoch
        global_ntrain        = ntrain

        # Calculate the ACTUAL mean loss across the entire cluster
        fit_loss_epoch   = global_fit_train_mse / global_ntrain
        mae_loss_epoch_t = global_mae_target / global_ntrain
        mse_loss_epoch_t = global_mse_target / global_ntrain

        # store epoch metrics:
        per_epoch_callback['epochs'].append(step)
        per_epoch_callback['loss'].append(fit_loss_epoch)
        per_epoch_callback['mae_target'].append(mae_loss_epoch_t)
        per_epoch_callback['mse_target'].append(mse_loss_epoch_t)

        # update scheduler:
        scheduler.step()

        if step % max(1, (cfg_train.max_epochs // 200)) == 0:
            logger.info(
                f'[Training] Epoch {step:03d}, Quantile loss: {fit_loss_epoch:.3f} - MAE target: {mae_loss_epoch_t:.3f}'
            )

        # plot epoch losses:
        if (step % cfg_tf_icl.callbacks.plot_freq == 0) or (step == cfg_train.max_epochs - 1):
            plot_losses(
                per_epoch_callback['epochs'],
                dict_losses = {'loss': per_epoch_callback['loss']},
                filename    = 'loss_epoch_{}'.format(cfg_tf_icl.data.name.lower()),
                save_path   = save_loss_path / 'epochs',
                clip_yaxis  = True
            )
            for l in ['mae', 'mse']:
                plot_losses(
                    per_epoch_callback['epochs'],
                    dict_losses = {
                        'target': per_epoch_callback['{}_target'.format(l)]
                    },
                    filename    = '{}_epoch_{}'.format(l, cfg_tf_icl.data.name.lower()),
                    save_path   = save_loss_path / 'epochs',
                    clip_yaxis  = True
                )

        # save best model:
        if fit_loss_epoch < best_loss:
            best_loss = fit_loss_epoch

            if cfg_train.enable_ckpt:
                model_to_save = tf_icl.state_dict()
                torch.save(
                    {
                        "data": cfg_tf_icl.data,
                        "cfg": cfg_tf_icl.tf_icl,
                        "epoch": step, 
                        "tf_icl": model_to_save, 
                        "optimizer": optimizer.state_dict(),
                        "train_loss": fit_loss_epoch, 
                        "global_step": global_steps # XXX Save AROMA here ??? XXX
                    },
                    save_ckpt_path / 'best.pt'
                )

    logger.info('[Training] end of training after {:,d} steps'.format(global_steps))

    return