from src.tools.inference.local.forecasting.naive import infer_fn as naive_infer_fn
from src.tools.inference.local.forecasting.prophet import infer_fn as prophet_infer_fn

__all__ = [
    "naive_infer_fn",
    "prophet_infer_fn"
]