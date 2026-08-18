from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

import torch


class LinearInterpolation:

    def __init__(self):
        pass

    def __call__(
        self,
        series_t: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Linear interpolation of missing values (mask == True)
        in a torch Tensor of shape `(N, T)`.
        
        Args:
            series_t (torch.Tensor): Value tensor of shape shape `(N, T)`, has NaNs.
            mask (torch.BoolTensor): Bool Tensor of shape `(N, T)`, True where `series_t` is missing

        Returns:
            series_interp (torch.Tensor): Tensor of shape `(N, T)`, with interpolated values.
        """

        return self.interpolate_series_torch(series_t, mask)

    @staticmethod
    def interpolate_series_torch(
        series_t: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Linear interpolation of missing values (mask == True)
        in a torch Tensor of shape `(N, T)`.
        
        Args:
            series_t (torch.Tensor): Value tensor of shape shape `(N, T)`, has NaNs.
            mask (torch.BoolTensor): Bool Tensor of shape `(N, T)`, True where `series_t` is missing

        Returns:
            series_interp (torch.Tensor): Tensor of shape `(N, T)`, with interpolated values.
        """

        series_np = series_t.cpu().numpy()
        mask_np = mask.cpu().numpy()

        N, T = series_np.shape
        t = np.arange(T)
        series_interp_np = np.copy(series_np)

        for i in range(N):
            observed = ~mask_np[i]
            if np.sum(observed) < 2:
                continue  # not enough points to run interpolation
            series_interp_np[i] = np.interp(t, t[observed], series_np[i][observed])
        
        return torch.tensor(series_interp_np, dtype=series_t.dtype, device=series_t.device).unsqueeze(-1)

class CubicSplineInteporlation:

    def __init__(self):
        pass
    
    def __call__(
        self,
        coords_t: torch.Tensor,
        series_t: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolation of missing values (mask == True) with cubic splines
        in a torch Tensor of shape `(N, T)`.
        
        Args:
            series_t (torch.Tensor): Value tensor of shape shape `(N, T)`, has NaNs.
            mask (torch.BoolTensor): Bool Tensor of shape `(N, T)`, True where `series_t` is missing

        Returns:
            series_interp (torch.Tensor): Tensor of shape `(N, T)`, with interpolated values.
        """

        return self.interpolate_series_spline_torch(coords_t, series_t, mask)
    
    @staticmethod
    def interpolate_series_spline_torch(
        coords_t: torch.Tensor,
        series_t: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpole les valeurs manquantes avec une spline cubique.
        series_t : (N, T)
        mask : (N, T) -> True où il manque des valeurs
        """

        coords_np = coords_t.cpu().numpy()
        series_np = series_t.cpu().numpy()
        mask_np = mask.cpu().numpy()
    
        N, T = series_np.shape
        # t = np.arange(T)
        series_interp_np = np.copy(series_np)
    
        for i in range(N):
            observed = ~mask_np[i]
            if np.sum(observed) < 2:
                continue  
    
            # Spline cubic
            f = interp1d(
                coords_np[i][observed], #t[observed],
                series_np[i][observed],
                kind       = 'cubic',      
                fill_value = "extrapolate" # type: ignore
            )
            # series_interp_np[i] = f(t)
            series_interp_np[i] = f(coords_np[i])
    
        return torch.tensor(series_interp_np, dtype=series_t.dtype, device=series_t.device).unsqueeze(-1)

class SeasonalNaive:

    def __init__(
        self,
        seasonality_in_days: int,
        nb_timesteps_per_day: int,
        max_lag_in_days: Optional[int] = None
    ):
        """
        Args:
            seasonality_in_days (int): main seasonality of the time series, *in days*
            nb_timesteps_per_day (int):
            max_lag_in_days (int): max lag tested, *in days* (if None: T // 2).
        """
        self.seasonality = int( seasonality_in_days * nb_timesteps_per_day )
        self.max_lag     = int( max_lag_in_days * nb_timesteps_per_day) if isinstance(max_lag_in_days, int) else None

    def __call__(
        self,
        series_t: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Impute missing values by copying observed values at lags ±24, ±48, ....
        If not available, use the mean over all observed values.

        Args:
            series_t (torch.Tensor): Value tensor of shape shape `(N, T)`, has NaNs.
            mask (torch.BoolTensor): Bool Tensor of shape `(N, T)`, True where `series_t` is missing

        Returns:
            series_filled (torch.Tensor): Tensor of shape `(N, T)`, with interpolated values.
        """

        return self.impute_by_offset( series_t, mask, step = self.seasonality, max_lag = self.max_lag )

    @staticmethod
    def impute_by_offset(
        series_t: torch.Tensor,
        mask: torch.Tensor,
        step: int = 24,
        max_lag: Optional[int] = None
    ) -> torch.Tensor:
        """
        Impute missing values by copying observed values at lags ±24, ±48, ....
        If not available, use the mean over all observed values.

        Args:
            series_t (torch.Tensor): Value tensor of shape shape `(N, T)`, has NaNs.
            mask (torch.BoolTensor): Bool Tensor of shape `(N, T)`, True where `series_t` is missing
            step (int, default: 24): time interval between two consecutive tries
            max_lag (int): max lag tested (if None: T // 2).

        Returns:
            series_filled (torch.Tensor): Tensor of shape `(N, T)`, with interpolated values.
        """
        
        N, T = series_t.shape
        if max_lag is None:
            max_lag = T // 2

        series_filled = series_t.clone()
        
        for n in range(N):
            # Moyenne des valeurs observées de la série n
            observed_values = series_t[n][~mask[n]]
            mean_value = observed_values.mean() if observed_values.numel() > 0 else torch.tensor(0.0, dtype=series_t.dtype, device=series_t.device)

            for t in range(T):
                if not mask[n, t]:
                    continue  # valeur déjà observée
                found = False
                for offset in range(step, max_lag + 1, step):
                    for direction in [-1, 1]:
                        neighbor_t = t + direction * offset
                        if 0 <= neighbor_t < T and not mask[n, neighbor_t]:
                            series_filled[n, t] = series_t[n, neighbor_t]
                            found = True
                            break
                    if found:
                        break
                if not found:
                    series_filled[n, t] = mean_value  # fallback par moyenne
        
        return series_filled.unsqueeze(-1)
