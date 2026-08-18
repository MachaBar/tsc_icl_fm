import torch



class LOCF:
    
    def __init__(
        self,
        horizon_in_days: int,
        nb_timesteps_per_day: int,
    ) -> None:
        
        self.horizon_len = int( horizon_in_days * nb_timesteps_per_day )

    def __call__(
        self,
        series_c: torch.Tensor
    ) -> torch.Tensor:
        return torch.repeat_interleave(series_c[:,-1:], self.horizon_len, dim=1)
    
class LastHorizonCF:

    def __init__(
        self,
        horizon_in_days: int,
        nb_timesteps_per_day: int,
    ) -> None:
        
        self.horizon_len = int( horizon_in_days * nb_timesteps_per_day )

    def __call__(
        self,
        series_c: torch.Tensor
    ) -> torch.Tensor:
        
        return series_c[:,-self.horizon_len:].clone()

class SeasonalNaive:

    def __init__(
        self,
        seasonality_in_days: int,
        horizon_in_days: int,
        nb_timesteps_per_day: int,
    ) -> None:
        
        self.horizon_len = int( horizon_in_days * nb_timesteps_per_day )
        self.seasonality = int( seasonality_in_days * nb_timesteps_per_day )
    
    def __call__(
        self,
        series_c: torch.Tensor
    ) -> torch.Tensor:
        
        long_horizon = self.horizon_len > self.seasonality
        n            = (self.horizon_len // self.seasonality) + (self.horizon_len % self.seasonality > 0)
        start_idx    = -int(self.seasonality * n) if long_horizon else -self.seasonality

        end_idx = start_idx + self.horizon_len
        if end_idx == 0:
            end_idx = None

        return series_c[:,start_idx:end_idx].clone()