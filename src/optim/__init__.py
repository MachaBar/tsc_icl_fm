from token import OP
from omegaconf import DictConfig

import torch.nn as nn
from torch.optim import AdamW, Optimizer

from src.optim.scheduler import get_scheduler
from src.optim.muon import Muon, split_into_param_groups_for_muon
from src.optim.soap import SOAP


def get_optimizer(
    model: nn.Module,
    cfg_optim: DictConfig,
    rank: int = 0
) -> Optimizer:

    # get remaining kwargs (can be different for each optimizer)
    kwargs = {k: v for k, v in cfg_optim.items() if k not in ['name', 'lr', 'weight_decay', 'lr_schedule']}
    
    name         = cfg_optim.name.lower()
    lr           = cfg_optim.get('lr', cfg_optim.get('lr_inr'))
    weight_decay = cfg_optim.get('weight_decay', 0.01)

    if name == 'adamw':
        opt = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    elif name == 'soap':
        opt = SOAP(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs # type: ignore
        )

    elif name == 'muon':
        param_groups = split_into_param_groups_for_muon(model)
        if rank == 0:
            print("Muon params group length:", len(param_groups[0]['params']), " / AdamW params group length:", len(param_groups[1]['params']))
        opt = Muon(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs # type: ignore
        )
    
    else:
        raise ValueError(f"Unknown optimizer name {name}.")
    
    return opt

__all__ = [
    "get_optimizer",
    "get_scheduler"
]