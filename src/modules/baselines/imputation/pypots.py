# pypots_imputer_factory.py
from __future__ import annotations

import importlib
import inspect
import os
import yaml
import logging

from typing import Dict, Tuple, Optional, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---- Registry 
# Map canonical model names to (module_path, class_name).
# Names sourced from current PyPOTS docs (2025-08-20).
REGISTRY: Dict[str, Tuple[str, str]] = {
    # Imputers
    "saits": ("pypots.imputation.saits", "SAITS"),
    "transformer": ("pypots.imputation.transformer", "Transformer"),
    "itransformer": ("pypots.imputation.itransformer", "iTransformer"),
    "imputeformer": ("pypots.imputation.imputeformer", "ImputeFormer"),
    "brits": ("pypots.imputation.brits", "BRITS"),
    "mrnn": ("pypots.imputation.mrnn", "MRNN"),
    "grud": ("pypots.imputation.grud", "GRUD"),
    "csdi": ("pypots.imputation.csdi", "CSDI"),
    "moment": ("pypots.imputation.moment", "MOMENT"),
    "locf": ("pypots.imputation.locf", "LOCF"),

    # Multitask models and forecasters (supported for imputation in PyPOTS)
    "timesnet": ("pypots.imputation.timesnet", "TimesNet"),
    "patchtst": ("pypots.imputation.patchtst", "PatchTST"),
    "etsformer": ("pypots.imputation.etsformer", "ETSformer"),
    "dlinear": ("pypots.imputation.dlinear", "DLinear"),
    "crossformer": ("pypots.imputation.crossformer", "Crossformer"),
    "micn": ("pypots.imputation.micn", "MICN"),
    "frets": ("pypots.imputation.frets", "FreTS"),
    "film": ("pypots.imputation.film", "FiLM"),
    "moderntcn": ("pypots.imputation.moderntcn", "ModernTCN"),
    "gpt4ts": ("pypots.imputation.gpt4ts", "GPT4TS"),
    "timellm": ("pypots.imputation.timellm", "TimeLLM"),
    "fits": ("pypots.imputation.fits", "FITS"),
    "tslanet": ("pypots.imputation.tslanet", "TSLANet"),
    "timemixer": ("pypots.imputation.timemixer", "TimeMixer"),
    "timemixerpp": ("pypots.imputation.timemixerpp", "TimeMixerPP"),
    "gpvae": ("pypots.imputation.gpvae", "GPVAE"),
    "totem": ("pypots.imputation.totem", "TOTEM"),
    "tefn": ("pypots.imputation.tefn", "TEFN"),
    "koopa": ("pypots.imputation.koopa", "Koopa"),
    "segrnn": ("pypots.imputation.segrnn", "SegRNN"),
    "fedformer": ("pypots.imputation.fedformer", "FEDformer"),
    "stemgnn": ("pypots.imputation.stemgnn", "StemGNN"),
}

ALIASES = {
    "sait": "saits",
    "cids": "csdi",
    "timenet": "timesnet",
    "i-transformer": "itransformer",
    "gpt-4ts": "gpt4ts",
    "time-llm": "timellm",
    "time_mixer": "timemixer",
    "time_mixer++": "timemixerpp",
}


def _canon(name: str) -> str:
    key = name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    # map alias keys (normalized the same way)
    alias_map = {k.replace("-", "").replace("_", ""): v for k, v in ALIASES.items()}
    return alias_map.get(key, key)


def available_imputers() -> Dict[str, Tuple[str, str]]:
    """Return the supported model registry."""
    return dict(REGISTRY)


def _import_class(module_path: str, class_name: str):
    mod = importlib.import_module(module_path)
    try:
        return getattr(mod, class_name)
    except AttributeError as e:
        raise ImportError(f"Class {class_name} not found in {module_path}") from e


def _filter_kwargs_for_cls(cls, params: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs that appear in the class __init__ signature."""
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys())
    # always allow **kwargs “catch-all”
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        return params
    return {k: v for k, v in params.items() if k in valid}


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is not installed; cannot load a YAML config.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"YAML config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_imputer(
    name: str,
    n_steps: Optional[int] = None,
    n_features: Optional[int] = None,
    *,
    config_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **overrides,
):
    """
    Model factory for *all* PyPOTS imputation models.

    Parameters
    ----------
    name : str
        Model name (case/format insensitive). Examples:
        'saits', 'transformer', 'itransformer', 'imputeformer', 'brits', 'mrnn',
        'grud', 'csdi', 'moment', 'timesnet', 'patchtst', 'etsformer', 'dlinear',
        'crossformer', 'micn', 'frets', 'film', 'moderntcn', 'gpt4ts', 'timellm',
        'fits', 'tslanet', 'timemixer', 'timemixerpp', 'gpvae', 'totem', 'locf',
        'tefn', 'koopa', 'segrnn', 'fedformer', 'stemgnn', etc.
    n_steps : Optional[int]
        Time steps per sample. Required for most NN models; ignored if constructor
        doesn’t accept it (e.g., LOCF).
    n_features : Optional[int]
        Feature dimension. Required for most NN models; ignored if not needed.
    config_path : Optional[str]
        Path to a YAML file providing default hyperparameters.
    config : Optional[dict]
        You can pass an already-parsed config dict instead of a file.
    **overrides :
        Any hyperparameters to override/extend the YAML config. These are
        filtered to the model’s constructor signature automatically.

    Returns
    -------
    imputer : object
        An instantiated PyPOTS imputer.
    """
    model_key = _canon(name)
    if model_key not in REGISTRY:
        raise ValueError(
            f"Unknown imputer '{name}'. Available: {', '.join(sorted(REGISTRY))}"
        )

    module_path, class_name = REGISTRY[model_key]
    cls = _import_class(module_path, class_name)

    # Load configuration (defaults + model-specific)
    cfg = {}
    if config is not None:
        cfg = dict(config)  # shallow copy
    else:
        cfg = _load_yaml(config_path)

    defaults: Dict[str, Any] = (cfg.get("defaults") or {})
    by_model: Dict[str, Any] = (cfg.get("models", {}).get(model_key) or {})

    # Core params (n_steps/n_features) are included if provided, but will be
    # filtered out later if the target constructor doesn’t accept them.
    core = {}
    if n_steps is not None:
        core["n_steps"] = n_steps
    if n_features is not None:
        core["n_features"] = n_features

    merged: Dict[str, Any] = {**defaults, **by_model, **core, **overrides}
    # Drop explicit None values to avoid overriding library defaults
    merged = {k: v for k, v in merged.items() if v is not None}

    # Filter against the actual __init__ signature
    filtered = _filter_kwargs_for_cls(cls, merged)

    # Construct
    return cls(**filtered)


# ---- Test and support
def print_supported() -> None:
    """Print the supported model keys and their classes."""
    logging.info("List of supported pypots imputers")
    for k, (m, c) in sorted(REGISTRY.items()):
        logging.info(f"{k:12s} -> {m}.{c}")

def test():
    """Print the supported model keys and their classes."""
    logging.info("List of supported pypots imputers")
    for k, (m, c) in sorted(REGISTRY.items()):
        logging.info(f"{k:12s} -> {m}.{c}")
    
    model = make_imputer('saits', 
                         n_steps=12, 
                         n_features=1, 
                         config_path="../../config/pypots_baselines_config/imputers.yaml")


if __name__ == "__main__":
    print_supported()
    # test()
