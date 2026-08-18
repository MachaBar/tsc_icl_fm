import logging

from collections import defaultdict

from omegaconf import DictConfig
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.dataloader.sampler import DistributedJointSampler
from src.modules.aroma import AROMAEncoderDecoderKL, AROMAEncoderDecoderICL

from src.optim import get_optimizer
from src.optim.scheduler import get_scheduler
import src.optim.losses as losses

from src.tools.utils.plot import plot_losses, plot_loss_steps
from src.tools.trainer.utils import unpack_batch, znorm_fit_transform, update_metrics

from src.tools.inference.aroma import infer_fn

def prepare_optim(
    cfg_optim: DictConfig,
    cfg_train: DictConfig,
    device: torch.device,
    len_train_loader: int,
    is_icl_training: bool = True,
):
    
    # (1/3) LAMBDA TARGET/CONTEXT NOT USED ANYMORE

    # get trade-off coeff in loss function (outer step):
    use_target_for_training = True if is_icl_training else cfg_optim.use_target_for_training
    lambda_target           = 1.0  if is_icl_training else cfg_optim.get('lambda_target', 1.0)
    auto_lambda_tuning      = getattr(cfg_optim, 'auto_lambda_tuning', False)
    if not use_target_for_training:
        lambda_target  = 0.0
        lambda_context = 1.0
    else:
        lambda_context = 1.0 - lambda_target
    assert lambda_target >= 0
    assert lambda_context >= 0

    # (2/3) LOSS FUNCTION

    # define loss function: 
    loss_type = cfg_optim.get('loss_type','mse').lower()

    if loss_type in ['mse', 'huber']:
        loss_fn    = losses.per_element_huber_fn if loss_type == 'huber' else losses.per_element_mse_fn
        median_idx = 0

    elif loss_type == 'quantile':
        # quantiles loss 
        quantiles = torch.linspace(
            cfg_optim.start_quantile, 
            cfg_optim.end_quantile, 
            cfg_optim.nb_quantiles
        ).to(device)
        
        loss_fn = losses.MultiQuantileSmoothPinballLoss(quantiles=quantiles, alpha=0.01)
        median_idx = cfg_optim.quantile_median

    # (3/3) MAX EPOCH OR MAX STEPS

    max_global_steps = cfg_train.get('max_steps', 0)
    if (max_global_steps is None) or (max_global_steps <=0):
        max_global_steps = np.inf
        schedule_on_steps = False
        max_epochs = cfg_train.max_epochs
    else:
        schedule_on_steps = True
        max_epochs = max_global_steps // len_train_loader + 1
    
    return loss_fn, median_idx, max_global_steps, max_epochs, schedule_on_steps, lambda_target, lambda_context


def train_fn(
    inr: AROMAEncoderDecoderKL | AROMAEncoderDecoderICL | DDP,
    train_loader: DataLoader,
    cfg: DictConfig,
    path_results_exp: Path,
    rank: int = 0,
    val_loader: DataLoader | None = None
) -> None:
    """
    Main function for training an INR model.

    Args:
        inr (AROMAEncoderDecoderKL): INR network to train
        train_loader (DataLoader): train dataloader
        cfg (DictConfig): yaml config (meta file with all dependencies)
        path_results_exp (Path): path where to export losses & checkpoints
        rank (int): for DDP
    """

    logger = logging.getLogger(__name__)

    is_dist = dist.is_initialized()
    device = next(inr.parameters()).device # Retrieve device from the DDP model

    # Get config:
    cfg_optim = cfg.optim
    cfg_train = cfg.trainer

    grid_to_monitor   = ['target']
    metric_to_monitor = ['mae']

    # Build directories to export losses and checkpoints:
    save_loss_path = path_results_exp / 'loss'
    save_ckpt_path = path_results_exp / 'ckpt'
    (save_loss_path / 'steps').mkdir(exist_ok=True,parents=True)
    (save_loss_path / 'epochs').mkdir(exist_ok=True,parents=True)
    save_ckpt_path.mkdir(exist_ok=True)

    # Move model to device:
    inr = inr.to(device)

    # check whether we train icl:
    is_icl_training = isinstance(inr.module, AROMAEncoderDecoderICL) if isinstance(inr, DDP) else isinstance(inr, AROMAEncoderDecoderICL)

    # Define optimizer:
    optimizer = get_optimizer(inr, cfg_optim, rank=rank)

    # Define scheduler:
    scheduler = get_scheduler(cfg_optim.scheduler, optimizer)

    # gradient clipping:
    apply_grad_clip = getattr(cfg_optim, 'apply_grad_clip', False)

    loss_fn, median_idx, max_global_steps, max_epochs, schedule_on_steps, lambda_target, lambda_context = prepare_optim(
        cfg_optim        = cfg_optim,
        cfg_train        = cfg_train,
        device           = device,
        len_train_loader = len(train_loader),
        is_icl_training  = is_icl_training
    )

    # get trade-off coeff in loss function (outer step):
    use_target_for_training = True if is_icl_training else cfg_optim.use_target_for_training
    
    # get asinh transform:
    apply_asinh_transform = inr.module.apply_asinh_transform if isinstance(inr, DDP) else inr.apply_asinh_transform
    transform     = torch.asinh if apply_asinh_transform else lambda x: x
    inv_transform = torch.sinh if apply_asinh_transform else transform

    best_loss = np.inf
    best_target = np.inf
    best_val = np.inf

    # alloc some space to save metrics:
    per_epoch_callback = defaultdict(list)
    per_step_callback  = defaultdict(list)

    global_steps = 0

    val_loss = []

    if rank == 0:
        logger.info('[Training] use target grid for training: {}, lambda target = {:}'.format(use_target_for_training, lambda_target))
        logger.info('[Training] Max number of epochs {:,d} / steps {:,d}'.format(max_epochs, 0 if max_global_steps==np.inf else max_global_steps))

    # Iterate through the epochs (one step = one pass of the train loader):
    for step in range(max_epochs):

        if global_steps > max_global_steps - 1:
            break   

        if is_dist and isinstance(train_loader.batch_sampler, DistributedJointSampler):
            # Crucial for the DistributedJointSampler to ensure proper shuffling per epoch
            train_loader.batch_sampler.set_epoch(step)

        # initialize per-epoch metrics:
        fit_loss_epoch, kl_train_epoch = 0, 0
        metrics_epoch = {'{}_{}'.format(metric_name, segment): 0. for segment in grid_to_monitor for metric_name in metric_to_monitor}

        ntrain = 0
        
        # Iterate through the train dataloader:
        for substep, batch in enumerate(train_loader):

            inr.train()

            # unpack batch:
            series_c, coords_c, series_t, coords_t, covar_c, covar_t = unpack_batch(batch, device=device)

            # z-normalize series (fit on context, transform on context + target) + filter std:
            series_c, coords_c, series_t, coords_t, covar_all = znorm_fit_transform(
                series_c       = series_c,
                coords_c       = coords_c,
                series_t       = series_t,
                coords_t       = coords_t,
                covar_c        = covar_c,
                covar_t        = covar_t,
                eps            = 1e-5,
                max_znorm      = 50,
                apply_std_mask = cfg_optim.apply_mask_during_training
            )
            
            # print('{}_epoch_{}_{}'.format(rank, step,substep), 
            #       'subgrid_len={}'.format(series_c.shape[1]+series_t.shape[1]),
            #       'obs_ratio={:.1f}'.format(series_c.shape[1]/(series_c.shape[1]+series_t.shape[1])),
            #       'ncovars={}, covar_len={}'.format(covar_all.shape[1],covar_all.shape[2]) if (covar_c is not None) else None,
            #       )

            n_samples = series_c.shape[0]
            if n_samples == 0:
                continue
            
            # build query coords:
            query_coords = torch.cat([coords_c, coords_t], dim=1) # [N, T_in+T_mis, 1]

            # TS-ICL forward pass (encoder + decoder):
            out, kl_loss = inr(
                series               = series_c,
                coords               = coords_c,
                target_coords        = query_coords,
                series_covar         = covar_all,
                coords_covar         = query_coords if (covar_t is not None) else coords_c if (covar_c is not None) else None,
                undo_asinh_transform = False
            )

            filename = 'data_{}_{}_{}.pt'.format(rank,step,substep)
            # if (covar_all is not None):
                # torch.save({
                #     'series_c' : series_c.cpu(),
                #     'series_t' : series_t.cpu(),
                #     'coords_c' : coords_c.cpu(),
                #     'coords_t' : coords_t.cpu(),
                #     'covar'    : covar_all.cpu(),
                #     # 'out' : out.cpu()
                # }, path_results_exp / filename)
            # continue

            # compute loss on obs and missing coords separately:
            yhat_context, yhat_target = None, out
            # yhat_context, yhat_target = out[:,:n_obs], out[:,n_obs:]

            if yhat_context is not None:
                train_loss_c = loss_fn(yhat_context, transform(series_c)).mean()
            train_loss_t = loss_fn(yhat_target, transform(series_t)).mean()

            # sometimes, the prior generates covariates with nan --> this ruins the whole training
            # quick fix: skip step if loss is nan
            if torch.isnan(train_loss_t):
                continue

            # get total train loss:
            if is_icl_training or yhat_context is None:
                fit_loss = train_loss_t
            else:
                fit_loss = lambda_context * train_loss_c + lambda_target * train_loss_t 
            loss = fit_loss # + kl_weight * kl_loss

            # do gradient descent wrt all parameters:
            optimizer.zero_grad()
            loss.backward()

            # optionally, do gradient clipping:
            if apply_grad_clip:
                nn.utils.clip_grad_value_(inr.parameters(), clip_value=1.)

            # optmizer step:
            optimizer.step()

            if schedule_on_steps:
                scheduler.step()
            
            # update train metrics:
            for segment in grid_to_monitor:
                yhat = yhat_context if segment == 'context' else yhat_target
                if yhat is None:
                    continue
                gt   = series_c if segment == 'context' else series_t
                for metric_name in metric_to_monitor:
                    per_step_callback, metrics_epoch = update_metrics(
                        yhat              = inv_transform(yhat[..., median_idx:median_idx+1]),
                        series            = gt,
                        metric_epoch      = metrics_epoch,
                        per_step_callback = per_step_callback,
                        key               = '{}_{}'.format(metric_name, segment)
                    )

            # store step losses:
            per_step_callback['loss'].append(loss.item())
            
            # update total loss:
            fit_loss_epoch += fit_loss.item() * n_samples
            kl_train_epoch += kl_loss.item() * n_samples
           
            ntrain       += n_samples
            global_steps += 1

            # plot losses (stepwise):
            if (global_steps % 500 == 0 or global_steps == int(max_epochs * len(train_loader)) - 1) and (rank == 0):
                plot_loss_steps(
                    per_step_callback['loss'],
                    filename   = 'loss_step_{}'.format(cfg.data.name.lower()),
                    save_path  = save_loss_path / 'steps',
                    clip_yaxis = True
                )
                [
                    plot_loss_steps(
                        per_step_callback['{}_{}'.format(l,segment)],
                        filename   = '{}_{}_step_{}'.format(l,segment,cfg.data.name.lower()),
                        save_path  = save_loss_path / 'steps',
                        clip_yaxis = True
                    ) for l in metric_to_monitor for segment in grid_to_monitor
                ]
            
            if global_steps > max_global_steps - 1:
                break            

        if (rank == 0) and step < 5:
            free, total = torch.cuda.mem_get_info(device)
            mem_used_mb = (total - free) / 1024 ** 2
            logger.info("Memory used: {:,d}/{:,d}".format(int(mem_used_mb),int(total/ 1024 ** 2)))

        # --- DDP SYNCHRONIZATION ---
        if is_dist:
            # Create a tensor to gather local statistics
            metrics = torch.tensor(
                [ntrain, fit_loss_epoch] + list(metrics_epoch.values()),
                device=device
            )
            # Global sum across all GPUs (All-Reduce)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            global_ntrain, global_fit_train_mse = metrics[0].item(), metrics[1].item()
            global_metrics = {k:metrics[i+2].item() for i,k in enumerate(metrics_epoch.keys())}
        else:
            global_fit_train_mse = fit_loss_epoch
            global_metrics = metrics_epoch
            global_ntrain = ntrain

        # Calculate the ACTUAL mean loss across the entire cluster, and store:
        fit_loss_epoch = global_fit_train_mse / global_ntrain
        per_epoch_callback['epochs'].append(step)
        per_epoch_callback['loss'].append(fit_loss_epoch)
        for key, val in global_metrics.items():
            per_epoch_callback[key].append( val / global_ntrain )

        # update scheduler:
        if not schedule_on_steps:
            scheduler.step()

        #  validation dataloader:
        if (val_loader is not None) and (rank == 0):
            inr.eval()
            val_loss.append( infer_fn(
                inr                   = inr.module if is_dist else inr, # type: ignore
                test_loader           = val_loader,
                path_results_exp      = path_results_exp / 'loss',
                make_plots            = False,
                quantile_median       = median_idx,
                validation_mode       = True
            ) )
            plot_losses(
                    per_epoch_callback['epochs'],
                    dict_losses = {'loss': val_loss},
                    filename    = 'val_loss',
                    save_path   = save_loss_path,
                    clip_yaxis  = False
                )
            if val_loss[-1] < best_val:
                best_val = val_loss[-1]
                model_to_save = inr.module.state_dict() if isinstance(inr, DDP) else inr.state_dict()
                torch.save(
                    {
                        "data": cfg.data,
                        "cfg_inr": cfg.model,
                        "epoch": step, 
                        "inr": model_to_save, 
                        "optimizer_inr": optimizer.state_dict(),
                        "train_loss": fit_loss_epoch, 
                        "global_step": global_steps
                    },
                    save_ckpt_path / 'best_val.pt'
                )
            pd.DataFrame({'val_loss': val_loss}).to_csv( save_loss_path / 'val_loss.csv')

        # --- LOGS AND CHECKPOINTS (Rank 0 only) ---
        if rank == 0:
            if step % max(1, (max_epochs // 200)) == 0:
                logger.info(
                    f'[Training] Epoch {step:03d}, Global loss: {fit_loss_epoch:.3f} - MAE target: {per_epoch_callback['mae_target'][-1]:.3f}'
                )

            # plot epoch losses:
            if (step % cfg_train.plot_freq == 0) or (step == max_epochs - 1):
                plot_losses(
                    per_epoch_callback['epochs'],
                    dict_losses = {'loss': per_epoch_callback['loss']},
                    filename    = 'loss_epoch_{}'.format(cfg.data.name.lower()),
                    save_path   = save_loss_path / 'epochs',
                    clip_yaxis  = True
                )
                for l in metric_to_monitor:
                    plot_losses(
                        per_epoch_callback['epochs'],
                        dict_losses = {
                            # 'context': per_epoch_callback['{}_context'.format(l)],
                            'target': per_epoch_callback['{}_target'.format(l)]
                        },
                        filename    = '{}_epoch_{}'.format(l, cfg.data.name.lower()),
                        save_path   = save_loss_path / 'epochs',
                        clip_yaxis  = True
                    )

            # save best model:
            if fit_loss_epoch < best_loss:
                best_loss = fit_loss_epoch
                if step > 10:
                    logger.info(
                        f'[Training] Epoch {step:03d} --> save best ckpt with loss: {fit_loss_epoch:.3f}'
                    )
                if cfg_train.enable_ckpt:
                    # Unwrapping DDP model to save the state_dict correctly
                    model_to_save = inr.module.state_dict() if isinstance(inr, DDP) else inr.state_dict()
                    torch.save(
                        {
                            "data": cfg.data,
                            "cfg_inr": cfg.model,
                            "epoch": step, 
                            "inr": model_to_save, 
                            "optimizer_inr": optimizer.state_dict(),
                            "train_loss": fit_loss_epoch, 
                            "global_step": global_steps
                        },
                        save_ckpt_path / 'best.pt'
                    )
        
    if rank == 0:
        logger.info('[Training] end of training after {:,d} steps'.format(global_steps))

    return