from typing import Tuple
import math
from functools import partial

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, LambdaLR


def get_scheduler(cfg, optimizer):

    name = cfg.name

    if name == 'CosineAnnealing':
        scheduler_func = CosineAnnealingLR
        scheduler = scheduler_func(optimizer, T_max = cfg.T_max, eta_min = cfg.eta_min)
    elif name == 'CosineAnnealingWarmRestarts':
        scheduler_func = CosineAnnealingWarmRestarts
        scheduler = scheduler_func(optimizer, T_0 = cfg.T_max, T_mult = getattr(cfg, 'T_mult', 1), eta_min = cfg.eta_min)
    elif name == 'CosineWithWarmup':
        scheduler_func = get_cosine_with_hard_restarts_schedule_with_warmup
        scheduler = scheduler_func(
            optimizer,
            num_warmup_steps   = cfg.num_warmup_steps,
            num_training_steps = cfg.T_max,
            num_cycles         = 1,
            last_epoch         = -1,
            eta_min            = 1e-1
        )
    else:
        raise NotImplementedError
    
    return scheduler

def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    last_epoch: int = -1,
    eta_min: float = 1e-4
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        eta_min=eta_min
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int,
    eta_min: float = 1e-4
):
    
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    if progress >= 1.0:
        return eta_min
    
    return eta_min + max(
        0.0, (1 - eta_min) * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))
    )


def context_target_schedule(
    current_step: int,
    num_training_steps: int,
    num_warmup_steps: int = 1_000,
    context_min: float = 0.2,
) -> Tuple[float, float]:
    """ Cosine schedule for lambda_context and lambda_target."""
    
    if current_step < num_warmup_steps:
        return 1.0, 0.0
    
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )

    if progress >= 1.0:
        return context_min, 1. - context_min

    lambda_context = context_min + max(
        0.0, (1 - context_min) * 0.5 * (1.0 + math.cos(math.pi * (progress % 1.0)))
    )

    return lambda_context, 1.0 - lambda_context
