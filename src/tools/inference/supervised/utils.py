from typing import Tuple
import time
import math

import numpy as np

import torch

from src.tools.inference.common import update_gluonts_metrics

def build_train_dict(
    train: np.ndarray,
    overlay_mask_ratio: float = 0.0,
    seed: int = 0
) -> dict:
    """PyPOTS train_set dict; optional random masking for self-supervision."""
    if overlay_mask_ratio <= 0.0:
        return {"X": train}
    rng = np.random.default_rng(seed)
    mask = rng.random(train.shape) < overlay_mask_ratio
    X = train.copy()
    X_ori = train.copy()
    X[mask] = np.nan
    return {"X": X, "X_ori": X_ori}


def build_val_dict(
    masked: np.ndarray,
    gt: np.ndarray
) -> dict:
    """PyPOTS val_set dict: {'X': masked, 'X_ori': ground_truth}."""
    return {"X": masked, "X_ori": gt}


def impute_batch(
    model,
    X_masked: np.ndarray,
    clear_cuda: bool = True,
) -> Tuple[np.ndarray, float]:
    
    Btotal, T, C = X_masked.shape

    def _pick_imputed_array(imp_out):
        # some PyPOTS models return dicts
        if isinstance(imp_out, dict):
            for k in ("imputed_X", "X_pred", "imputed_data", "imputation", "imputed"):
                if k in imp_out:
                    imp_out = imp_out[k]
                    break
        if torch.is_tensor(imp_out):
            imp_out = imp_out.detach().cpu().numpy() # type: ignore
        return imp_out

    def _coerce_to_BTC(arr: np.ndarray, xb_shape: Tuple[int, int, int]) -> np.ndarray:
        """Coerce various model-specific layouts to [B, T, C]."""
        B, T, C = xb_shape
        sh = tuple(arr.shape)

        # already good
        if sh == xb_shape:
            return arr

        # common variants
        if sh == (B, T):                 # [B, T] -> add channel
            return arr[..., None]
        if sh == (B, T, 1):              # already [B, T, 1]
            return arr
        if sh == (B, 1, T):              # [B, 1, T] -> [B, T, 1]
            return np.transpose(arr, (0, 2, 1))
        if sh == (B, 1, T, 1):           # CSDI-style -> squeeze sample dim
            return np.squeeze(arr, axis=1)   # [B, T, 1]
        if sh == (B, C, T) and C == 1:   # [B, C, T] -> [B, T, C]
            return np.transpose(arr, (0, 2, 1))
        if sh == (B, 1, T, C):           # [B, 1, T, C] -> squeeze axis 1
            return np.squeeze(arr, axis=1)
        if sh == (B, T, C):              # already correct (generic)
            return arr

        # generic rescue: find T and C axes (keep batch at 0)
        if arr.ndim >= 3 and sh[0] == B:
            try:
                t_axis = next(i for i, s in enumerate(sh) if i != 0 and s == T)
                # pick a channel axis different from t_axis
                c_candidates = [i for i, s in enumerate(sh)
                                if i not in (0, t_axis) and (s == C or s == 1)]
                if not c_candidates:
                    raise ValueError("no channel axis candidate")
                c_axis = c_candidates[0]
                out = np.moveaxis(arr, (t_axis, c_axis), (1, 2))
                # squeeze/drop extras if present
                out = out.reshape(B, T, -1)
                if out.shape[2] != C:
                    # if model produced >1 channels but we expect 1, take the first
                    out = out[:, :, :C]
                return out
            except Exception:
                pass

        raise RuntimeError(
            f"{type(model).__name__}.impute() returned shape {sh}; "
            f"couldn't coerce to {xb_shape}."
        )

    xb = X_masked
 
    # timing only the impute call
    if clear_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    # try to pass batch_size if supported
    try:
        imp_out = model.impute({"X": xb}, batch_size=Btotal)
    except TypeError:
        imp_out = model.impute({"X": xb})
    if clear_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time_s = (time.perf_counter() - t0)

    imp_b = _pick_imputed_array(imp_out)
    imp_b = _coerce_to_BTC(imp_b, xb.shape) # type: ignore
    
    return imp_b, inference_time_s


def impute_and_score_streaming(
    model,
    X_masked: np.ndarray,
    X_gt: np.ndarray,
    dict_evaluators: dict,
    batch_size: int = 64,
    clear_cuda: bool = True,
) -> Tuple[float, float, float, float, float, int, dict]:
    """
    Batch-wise imputation and error on missing positions where GT is known.
    Returns:
      (mse_mean, mae_mean, mse_std, mae_std, inference_time_s)
    Std values are computed across all evaluated missing positions.
    """
    assert X_masked.shape == X_gt.shape and X_masked.ndim == 3
    Btotal, T, C = X_masked.shape

    # accumulators for mean & std via sum/sum of squares
    # for MSE we accumulate squared error (se); for MAE we accumulate absolute error (ae)
    se_sum = 0.0
    se_sqsum = 0.0
    ae_sum = 0.0
    ae_sqsum = 0.0
    cnt = 0
    nb_chunks = 0

    inference_time_s = 0.0

    assert len(dict_evaluators) == 1
    model_name = list(dict_evaluators.keys())[0]

    for s in range(0, Btotal, batch_size):
        e = min(Btotal, s + batch_size)
        xb = X_masked[s:e]
        gb = X_gt[s:e]

        nb_chunks += len(gb)

        imp_b, infer_time_b = impute_batch( model, xb, clear_cuda=clear_cuda )
        inference_time_s   += infer_time_b

        miss = np.isnan(xb)
        gt_ok = ~np.isnan(gb)
        eval_mask = miss & gt_ok

        dict_evaluators[model_name] = update_gluonts_metrics(
            ytrue           = torch.Tensor(gb[eval_mask]).reshape(len(eval_mask),-1,1),
            yhat            = torch.Tensor(imp_b[eval_mask]).reshape(len(eval_mask),-1,1),
            evaluators      = dict_evaluators[model_name]
        )
        if eval_mask.any():
            diff = (imp_b - gb)[eval_mask]
            ae = np.abs(diff).astype(np.float64)
            se = (diff * diff).astype(np.float64)

            # accumulate sums for mean/std
            ae_sum += float(np.sum(ae))
            ae_sqsum += float(np.sum(ae * ae))
            se_sum += float(np.sum(se))
            se_sqsum += float(np.sum(se * se))
            cnt += int(eval_mask.sum())

        if clear_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    if cnt == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), inference_time_s, dict_evaluators

    mse_mean = se_sum / cnt
    mae_mean = ae_sum / cnt
    # std across elements: sqrt(E[x^2] - (E[x])^2)
    mse_var = max(0.0, (se_sqsum / cnt) - (mse_mean ** 2))
    mae_var = max(0.0, (ae_sqsum / cnt) - (mae_mean ** 2))
    mse_std = math.sqrt(mse_var)
    mae_std = math.sqrt(mae_var)

    return mse_mean, mae_mean, mse_std, mae_std, inference_time_s, nb_chunks, dict_evaluators
