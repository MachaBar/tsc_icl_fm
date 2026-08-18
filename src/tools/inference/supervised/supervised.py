import logging
from typing import Dict, Optional
import time
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

import pandas as pd

import torch
from torch.utils.data import Dataset

from pypots.utils.random import set_random_seed

from src.modules.baselines.imputation import make_imputer
from src.tools.inference.supervised.utils import build_train_dict, build_val_dict, impute_and_score_streaming
from src.tools.inference.common import initialize_gluonts_metrics

def run_one_imputer(
    imputer_name: str,
    dict_datasets: Dict[str, Dataset],
    cfg: DictConfig,
    path_results_exp: Path,
    *,
    epochs: int,
    batch_size: int,
    device: str,
    overlay_mask_ratio: float = 0.0,
    imputer_kwargs: Optional[dict] = None,
) -> None:
    """Fit one imputer and score all four test scenarios."""

    logger = logging.getLogger(__name__)

    if dict_datasets["train"] is not None:
        n_steps = dict_datasets["train"].shape[1]
        train_set = build_train_dict(dict_datasets["train"], overlay_mask_ratio=overlay_mask_ratio)

        if dict_datasets["val_gt"] is not None:
            val_set = build_val_dict(dict_datasets["val"], dict_datasets["val_gt"])  # choose bm1 for val
        else:
            val_set = build_train_dict(dict_datasets["train"], overlay_mask_ratio=0.2)

    else:
        n_steps = None
        train_set = None
        val_set = None

    # Each chunk is univariate after Extract_data -> [N, T, 1]
    model = make_imputer(
        imputer_name,
        n_steps     = n_steps,
        n_features  = 1,
        batch_size  = batch_size,
        epochs      = epochs,
        device      = device,
        config      = OmegaConf.to_container(cfg.supervised),
        **(imputer_kwargs or {}),
    )
    
    model_name = model.__class__.__name__
    logger.info('[Model Prep] Sucessfully instantiated supervised model of class {}'.format(model_name))
    logger.info('[Model Prep] End of model prep.\n')

    print(imputer_kwargs)

    set_random_seed(random_seed=int( imputer_kwargs.get('random_seed', 2025) ))

    # --- training time ---
    logger.info('[Training] Start training {}...'.format(model_name))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.fit(train_set=train_set, val_set=val_set)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_time_s = time.perf_counter() - t0
    logger.info('[Training] End of training.\n')

    # inference

    test_scenarios = dict_datasets['test']

    rows = []
    for sc, x_test_dict in test_scenarios.items():

        dict_evaluators = {model_name:initialize_gluonts_metrics(axis=None)}

        x_masked = x_test_dict.get('masked')
        x_gt     = x_test_dict.get('gt')

        mse, mae, mse_std, mae_std, infer_s, nb_chunks, dict_evaluators = impute_and_score_streaming(
            model           = model,
            X_masked        = x_masked,
            X_gt            = x_gt,
            dict_evaluators = dict_evaluators,
            batch_size      = batch_size
        )

        save_dir_metrics: Path = path_results_exp / sc / 'inference' / 'metrics'
        save_dir_metrics.mkdir(exist_ok=True, parents=True)

        # build dict of aggregated metrics and export:
        names = list( dict_evaluators[model_name].keys() )
        metrics = pd.DataFrame( {
            'Model'  : [model_name],
            'Chunks' : [nb_chunks]
        } | {
            name:[dict_evaluators[k][name].get().mean() for k in dict_evaluators.keys()] for name in names
        } | {
            'Time (s)' : [training_time_s + infer_s]
        } )
        metrics.set_index('Model', inplace=True)
        metrics.to_csv( save_dir_metrics / 'gluonts_metrics.csv' )
    
    return


