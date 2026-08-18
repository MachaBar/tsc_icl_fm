from typing import Optional, List, Tuple, Literal, Dict

import numpy as np
import torch
from einops import rearrange, repeat

from src.modules.aroma import AROMAEncoderDecoderICL
from src.modules.aroma.aroma.encoder import PerceiverEncoder, UnivariatePerceiverEncoder
from src.modules.icl_learning import ICLearning, ICLearningCovar, ICLearningCrossAttn

from src.data.scaler import CustomStandardScaler

class TSICL(AROMAEncoderDecoderICL):
    """
    Main interface to interact with TS-ICL.

    Shared for the imputation and forecasting settings.

    Parameters
    ----------

    encoder : PerceiverEncoder | UnivariatePerceiverEncoder
        Perceiver-based time series encoder
    
    head : ICLearning | ICLearningCovar | ICLearningCrossAttn
        Transformer head performing In-Context Learning regression
    
    apply_asinh_transform : bool
        Whether to add `asinh` transform after input z-normalization.
    
    setting : Literal['forecasting', 'imputation']
        Activate the `forecasting` or `imputation` inference mode

    max_context_len : int
        Max context length

    max_target_len : int
        Max horizon length (forced to 0 in case of imputation)
    """

    def __init__(
        self,
        encoder: PerceiverEncoder | UnivariatePerceiverEncoder,
        head: ICLearning | ICLearningCovar | ICLearningCrossAttn,
        apply_asinh_transform: bool,
        setting: Literal['imputation', 'forecasting'],
        max_context_len: int = 4096,
        max_target_len: int = 672
    ):
        super().__init__(encoder, head, apply_asinh_transform)
        
        # task config:
        self.is_grid_absolute = True
        self.do_forecasting   = setting == 'forecasting'

        start_quantile = self.tf_icl.start_quantile
        end_quantile   = self.tf_icl.end_quantile
        nb_quantiles   = self.tf_icl.nb_quantiles

        self.training_quantile_levels = np.linspace(
            start_quantile * 100, 
            end_quantile * 100, 
            nb_quantiles
        ).tolist()

        self.max_context_length = max_context_len
        self.max_target_length  = max_target_len if self.do_forecasting else 0

    # ---------------
    # CORE UTILS
    # ---------------
    
    def _get_grid_len( self ) -> int:
        """ Get the length of the grid used by TS-ICL"""

        # get grid length used during training:
        train_grid_length  = self.max_context_length + self.max_target_length
        
        return train_grid_length

    @staticmethod
    def make_grid(
        grid_length: int,
        num_samples: int,
        end_point: float = 1.0,
        start_point: float = 0.0
    ) -> torch.Tensor:
        """
        Create a grid of time coords in (0, 1)

        Args:
            grid_length (int): number of timesteps
            num_samples (int): number of samples
            end_point (float): end point of the grid
            start_point (float): start point of the grid
        
        Returns:
            Grid as torch.Tensor of shape `(num_samples, grid_length, 1)`.
        """
        
        grid = torch.linspace(start_point, end_point, grid_length).float().unsqueeze(0).repeat_interleave(repeats=num_samples, dim=0)
        grid = grid.unsqueeze(-1) # [N, T, 1]
        return grid
    
    @staticmethod
    def complete_nans(
        X: torch.Tensor,
        grid: torch.Tensor,
        is_test: bool = False,
        X_cov: torch.Tensor | None = None
    ) -> Dict[str, torch.Tensor]:
        """
        Fills missing values (NaNs) in the dataset (replaced by observed values at random instants of the same series).

        Args:
            X (torch.Tensor): value tensor of shape `(bs, seq_len, 1)`
            grid (torch.Tensor): grid coord tensor of shape `(bs, seq_len, 1)`
            is_test (bool): True if test mode
        
        Returns:
            Value and coord tensors of shape `(bs, seq_len, 1)` with duplicates instead of NaNs
        """

        has_nans = torch.isnan(X).any().item()
        
        if not has_nans:

            values_with_replacement = X.clone()
            grids_with_replacement  = grid.clone()
            cov_with_replacement    = X_cov

        else:
            
            if is_test:
                torch.manual_seed(42)

            # prepare tensors:
            has_batch_dim = X.ndim == 3
            X    = X.clone() if has_batch_dim else X.clone().unsqueeze(0)
            grid = grid.clone() if has_batch_dim else grid.clone().unsqueeze(0)

            mask_observed = ~torch.isnan(X[..., 0]) # [N, T]
            X_filled      = X[..., 0].clone()       # [N, T]
            partial_grid  = grid[..., 0].clone()    # [N, T]

            if X_cov is not None:
                X_cov = X_cov.clone() if has_batch_dim else X_cov.clone().unsqueeze(0)
                X_cov_filled = X_cov[..., 0].clone()

            # loop through samples:
            for i in range(X_filled.shape[0]):

                # time indices with no NaNs in raw data:
                valid_indices = torch.where(mask_observed[i])[0]

                # time indices corresponding to NaNs in raw data:
                invalid_indices = torch.where(~mask_observed[i])[0]
                
                # replace NaNs by any other observed value:
                if valid_indices.numel() > 0 and invalid_indices.numel() > 0:  
                    replacements = valid_indices[torch.randint(0, valid_indices.numel(), (invalid_indices.numel(),))]
                    X_filled[i, invalid_indices]     = X[i, replacements, 0]
                    partial_grid[i, invalid_indices] = grid[i, replacements, 0]
                    if X_cov is not None:
                        X_cov_filled[i, :, invalid_indices] = X_cov[i, :, replacements, 0]
            
            values_with_replacement = X_filled.unsqueeze(-1)      # [N, T, 1] (with duplicates)
            grids_with_replacement  = partial_grid.unsqueeze(-1)  # [N, T, 1] (with duplicates)
            if X_cov is not None:
                cov_with_replacement = X_cov_filled.unsqueeze(-1) # [N, T, 1] (with duplicates)

            if (not has_batch_dim):
                values_with_replacement = values_with_replacement.squeeze(0)
                grids_with_replacement  = grids_with_replacement.squeeze(0)
                if X_cov is not None:
                    cov_with_replacement = cov_with_replacement.squeeze(0)

        # if torch.isnan(values_with_replacement).any():
        #     values_with_replacement = torch.nan_to_num(values_with_replacement, nan=0.0)

        out = {
            'values' : values_with_replacement,
            'coords' : grids_with_replacement,
        }
        if X_cov is not None:
            assert isinstance(cov_with_replacement, torch.Tensor)
            out['covar'] = cov_with_replacement
            
        return out

    # ---------------
    # ADAPTER UTILS
    # ---------------

    # XXX TODO fevbench past-only covar

    def _check_input_dim(
        self,
        x: torch.Tensor,
        has_batch_dim: bool = True
    ) -> torch.Tensor:
        """
        Convert arbitrary tensors of C channels and T timesteps into 3D univariate tensors.

        Args:
            x (torch.Tensor): input tensor to check
            has_batch_dim (bool): whether `x` has a first batch dimension
        
        Returns:
            3D Tensor of shape `(bs x C, T, 1)` (variable dim in the batch dim)
        """
        
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(-1)
        elif x.ndim == 2:
            dim = -1 if has_batch_dim else 0
            x = x.unsqueeze(dim) # (b, t, 1)

        assert x.ndim == 3

        self.num_var = x.shape[-1]
        if self.num_var > 1:
            x = rearrange(x, 'b t c -> (b c) t 1')
        
        return x
    
    def _batch_inputs_utils(
        self,
        inputs: list | torch.Tensor,
        batch_size: int
    ) -> Tuple[int, bool, list | torch.Tensor | dict]:
        
        # convert list of tensors to stacked tensor:
        if isinstance(inputs, list):

            # for fevbench
            if isinstance(inputs[0], dict):
                past_covars, future_covars = None, None
                self._future_cov_keys = []
                target = self._list_dict_inputs_utils(inputs, key = 'target')
                
                if 'future_covariates' in inputs[0]:
                    future_covars = self._list_dict_inputs_utils(inputs, key = 'future_covariates')
                if 'past_covariates' in inputs[0]:
                    past_covars = self._list_dict_inputs_utils(inputs, key = 'past_covariates')
                # target, past_covars, future_covars are Tensor or List[Tensor]
                
                out = {
                    'target'  : target,
                    'covar_c' : past_covars,
                    'covar_t' : future_covars
                }

                # assert all(['target' in val for val in inputs])
                # if all([val['target'].shape == inputs[0]['target'].shape for val in inputs]):
                #     inputs = torch.stack([torch.Tensor(x['target']) for x in inputs], dim=0) # (bs, t, 1)
                # else:
                #     inputs = [x['target'] for x in inputs] # [(t, 1), (t, 1), ...]

            # for time
            elif isinstance(inputs[0], torch.Tensor):
                if all([val.shape == inputs[0].shape for val in inputs]):
                    out = torch.stack(inputs, dim=0)
                else:
                    out = inputs

        else:
            out = inputs

        # case 1: batched tensor:
        is_tensor = True
        if isinstance(out, torch.Tensor):
            out      = self._check_input_dim(out, has_batch_dim = True) # (bs, t, 1)
            num_batches = len(out) // batch_size + (len(out) % batch_size > 0)

        # case 2: list of 1D tensors (irregular lengths):
        elif isinstance(out, list):
            out = [self._check_input_dim(x, has_batch_dim = False) for x in out] # [(1, t, 1), (1, t, 1), ...]
            num_batches = len(out)
            is_tensor = False

        elif isinstance(out, dict):

            assert ('target' in out) #and isinstance(out['target'], torch.Tensor)
            # assert ('covar_c' in out) and (out['covar_c'] is None or isinstance(out['covar_c'], torch.Tensor))
            # assert ('covar_t' in out) and (out['covar_t'] is None or isinstance(out['covar_t'], torch.Tensor))

            if isinstance(out['target'], torch.Tensor):
                out['target'] = self._check_input_dim(out['target'], has_batch_dim=True)
                num_batches = len(out['target']) // batch_size + (len(out['target']) % batch_size > 0)
            elif isinstance(out['target'], list):
                out['target'] = [self._check_input_dim(x, has_batch_dim = False) for x in out['target']]
                is_tensor = False
                num_batches = len(out['target'])
            else:
                raise NotImplementedError        

        else:
            raise NotImplementedError
                
        return num_batches, is_tensor, out
    
    def _list_dict_inputs_utils(
        self,
        list_inputs: list[dict],
        key: str
    ):

        first_input = list_inputs[0]
        assert isinstance(first_input[key], dict) or isinstance(first_input[key], torch.Tensor)        
        assert all([key in d for d in list_inputs])

        # handle 'target':
        if isinstance(first_input[key], torch.Tensor):
            if all([d[key].shape == list_inputs[0][key].shape for d in list_inputs]):
                out = torch.stack([torch.Tensor(d[key]) for d in list_inputs]) # (bs, ...)
            else:
                out = [d[key] for d in list_inputs]

        # handle 'past_covariates' and 'future_covariates'
        else:
            list_keys = list(first_input[key].keys())
            
            if len(list_keys) == 0:
                return None
            
            if len(self._future_cov_keys) > 0:
                list_keys = self._future_cov_keys.copy()
            if key == 'future_covariates':
                self._future_cov_keys = list_keys
            assert all([all([k in d[key] for k in list_keys]) for d in list_inputs])      
            out = [
                torch.stack(
                    [
                        torch.Tensor(d[key][k]) for k in list_keys
                    ],
                    dim = 0 # (c t 1)
                ) for d in list_inputs
            ]
            if all([x.shape == out[0].shape for x in out]):
                out = torch.stack(out, dim = 0) # (bs, c, t, 1)

        return out
    
    def _batch_covar_utils(
        self,
        covars: List[torch.Tensor] | torch.Tensor | None
    ) -> Tuple[bool, List[torch.Tensor] | torch.Tensor | None]:
        
        if covars is None:
            return True, None

        if isinstance(covars, list): # List[torch.Tensor]
            assert isinstance(covars[0], torch.Tensor)
            if all([val.shape == covars[0].shape for val in covars]):
                covars = torch.stack(covars, dim=0) # (bs, T, c)

        # case 1: batched tensor:
        is_cov_tensor = True
        if isinstance(covars, torch.Tensor):
            if covars.ndim == 3:
                covars      = rearrange(covars, 'b t c -> b c t 1') # (bs, C, t, 1)
            elif covars.ndim == 4:
                assert covars.shape[-1] == 1
            # assert len(covars) == len(inputs)

        # case 2: list of 1D tensors (irregular lengths):
        elif isinstance(covars, list):
            if covars[0].ndim == 2:
                covars = [rearrange(x, 't c -> 1 c t 1') for x in covars] # [(1, c, t, 1), (1, c, t, 1), ...]
            elif covars[0].ndim == 3:
                assert covars[0].shape[-1] == 1
                covars = [rearrange(x, 'c t 1 -> 1 c t 1') for x in covars] # [(1, c, t, 1), (1, c, t, 1), ...]
            is_cov_tensor = False
            # assert isinstance(inputs, list)
            # assert len(covars) == len(inputs)

        return is_cov_tensor, covars

    def input_adapter(
        self,
        inputs,
        covars: List[torch.Tensor] | torch.Tensor | None = None,
        batch_size: int = 32
    ):
        """
        **STILL ON-GOING DEV**

        Take input series of different formats to standardized TS-ICL inputs.

        *Supports inputs from: `fevbench`, `fevbench_covar`, `time`, `time_covar`
        for `imputation` and `forecasting` tasks.*
        """
        
        # covar setting:
        has_covar = covars is not None
        covar_c   = None
        covar_t   = None

        # prepare inputs:
        num_batches, is_tensor, inputs = self._batch_inputs_utils(inputs, batch_size=batch_size)
        if isinstance(inputs, dict):
            covar_c = inputs['covar_c']
            covar_t = inputs['covar_t']
            inputs  = inputs['target']
            # these are Tensor or List[Tensor]
            has_covar = (covar_c is not None)
            if has_covar:
                covars = covar_c if covar_t is None else torch.cat(
                    [covar_c, covar_t], dim = -2
                ) if is_tensor else [
                    torch.cat([x_c, x_t], dim=-2) for x_c, x_t in zip(covar_c,covar_t)
                ]
                # (bs, c, t, 1)
        # inputs is now a tensor (bs, t, 1) or a list of tensors (1, t, 1)

        # covars:
        is_cov_tensor, covars = self._batch_covar_utils(covars)
        if covars is not None:
            # covars is now a tensor (bs, c, t, 1) or a list of tensors (c, t, 1)
            if len(covars) != len(inputs):
                assert int(len(covars)) == 1, print('covars {} vs inputs {}'.format(covars.shape, inputs.shape)) # pyright: ignore[reportAttributeAccessIssue]
                covars = repeat(covars, 'n c t 1 -> (n b) c t 1',b=len(inputs))
            assert len(covars) == len(inputs)
            assert is_cov_tensor == is_tensor

        return inputs, covars, num_batches, is_tensor, has_covar

    # ---------------
    # TASK UTILS
    # ---------------

    # XXX can be factorized between imputation & forecasting

    def _get_context_target_coords_f(
        self,
        grid: torch.Tensor,
        series_c: torch.Tensor,
        horizon_len: int,
        covariates: torch.Tensor | None = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract context and target tensors of the forecasting task.

        Args:
            grid (torch.Tensor): raw time grid of shape `(bs, T_grid, 1)`
            series_c (torch.Tensor): tensor of past values of shape `(bs, T, 1)`
            horizon_len (int): prediction horizon (target)
        
        Returns:
            Values on the lookback, time coords on the lookback and the query points.
        """
        
        has_covar       = isinstance(covariates, torch.Tensor)
        covar_on_future = False

        lookback_len = min(series_c.shape[1], self.max_context_length)

        grid_threshold = self.max_context_length
        coords_c = grid[:,(grid_threshold-lookback_len):grid_threshold]
        coords_t = grid[:,grid_threshold:(grid_threshold+horizon_len)]

        # handle covar:
        if has_covar:
            covar_on_future = covariates.shape[-2] > series_c.shape[-2]
            if covar_on_future:
                assert covariates.shape[-2] == horizon_len + series_c.shape[-2]

            covar_t = covariates[..., -horizon_len:, :] if covar_on_future else None
            covariates_c = covariates[..., :-horizon_len, :] if covar_on_future else covariates
            covariates_c = covariates_c[..., -lookback_len:,:]
            assert covariates_c is not None
        else:
            covariates_c, covar_t = None, None

        # make sure we don't exceed max context len:
        series_c = series_c[:,-lookback_len:]

        # get the total number of missing points in each sample:
        nb_missing_points = torch.isnan(series_c).sum(1).squeeze() # (bs)
        if nb_missing_points.ndim == 0:
            nb_missing_points = nb_missing_points.unsqueeze(0)
        
        # compute the maximum context length:
        max_context_len   = int( series_c.shape[1] - min(nb_missing_points) )

        # prepare context tensors; handle each sample separately:
        list_coords_c, list_series_c, list_covar_c = [], [], []
        for idx in range(len(series_c)):
            
            # get sample:
            sample = series_c[idx].squeeze()

            # get missing mask
            is_missing_mask = torch.isnan(sample) 

            # handle series that have too many nans (= more than the min nb of nan in the batch)
            if is_missing_mask.sum() > min(nb_missing_points):

                # get number of values actually observed:
                cur_len = len(sample[~is_missing_mask])
                mis_len = max_context_len - cur_len

                # (i) pad context values with nans, up to length=max_context_len:
                # (nan values will be discared when we complete_nans)
                series_i = torch.cat([
                    sample[~is_missing_mask],
                    torch.nan * torch.ones(mis_len).to(sample.device)
                ]).unsqueeze(-1) # (max_context_len, 1)

                # (ii) same padding (with ones) for the grid:
                grid_i = torch.cat([
                    coords_c[idx][~is_missing_mask].squeeze(),
                    torch.ones(mis_len).to(sample.device) # (T, 1), (T, 1)
                ]).unsqueeze(-1) # (max_context_len, 1)

                # (iii) handle covariates (context part):
                if has_covar:

                    # check time alignment:
                    assert isinstance(covariates_c, torch.Tensor)
                    assert covariates_c.shape[-2] == series_c.shape[-2]
                    
                    # we keep only the covariates observed when the target series is observed
                    # + then we must pad with nans:
                    nb_covars = covariates_c.shape[1] # (bs C t 1)
                    covar_c = torch.cat([
                        covariates_c[idx][...,~is_missing_mask,:], # (c t_obs 1)
                        torch.nan * torch.ones((nb_covars, mis_len, 1)).to(sample.device)
                    ], dim = - 2) # (C, max_context_len, 1)

                    # now, each covar may also contain NaN (and they may not be aligned between themselves)
            
                # (iv) complete nans:
                out = self.complete_nans(series_i, grid_i, is_test=True, X_cov=covar_c if has_covar else None)
                series_i, grid_i, covar_c = out['values'], out['coords'], out.get('covar')

            else:
                series_i, grid_i = series_c[idx][~is_missing_mask], coords_c[idx][~is_missing_mask] # (max_context_len, 1), (max_context_len, 1)
                if has_covar:
                    covar_c = covariates_c[idx][...,~is_missing_mask,:] # pyright: ignore[reportOptionalSubscript] # (C, max_context_len, 1)
                    assert covar_c.shape[-2] == series_i.shape[-2]
            
            # check that there is no nan in the context values:
            assert (not torch.isnan(series_i).any()) or torch.isnan(series_i).all()
            list_coords_c.append(grid_i.reshape(-1,1))   # (max_context_len, 1)
            list_series_c.append(series_i.reshape(-1,1)) # (max_context_len, 1)
            if has_covar:
                list_covar_c.append(covar_c)

        coords_c = torch.stack(list_coords_c) # (bs, max_context_len, 1)
        series_c = torch.stack(list_series_c) # (bs, max_context_len, 1)
        if has_covar:
            covariates_c = torch.stack(list_covar_c)    # (bs, C, max_context_len, 1)

        out = {
            'series_c' : series_c,
            'coords_c' : coords_c,
            'coords_t' : coords_t,
        }

        if has_covar:
            assert isinstance(covariates_c, torch.Tensor)
            out['covar_c'] = covariates_c # (bs, C, max_context_len, 1)
        if covar_on_future:
            assert isinstance(covar_t, torch.Tensor)
            out['covar_t'] = covar_t # (bs, C, query_seq_len, 1)


        return out
    
    def _get_context_target_coords_i(
        self,
        grid: torch.Tensor,
        series_c: torch.Tensor,
        covariates: torch.Tensor | None = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract context and target tensors of the imputation task.

        Args:
            grid (torch.Tensor): raw time grid of shape `(bs, T_grid, 1)`
            series_c (torch.Tensor): tensor of past values of shape `(bs, T, 1)`
            covariates (torch.Tensor, optional): tensor of covariates of shape `(bs, c, T, 1)`
        
        Returns:
            Values on the context, time coords on the context and the query points.
        """

        # grid (bs, T, 1)
        seq_len   = series_c.shape[1]
        grid      = grid[:,:seq_len,:]
        has_covar = covariates is not None
        covar_c   = None
        
        bs = len(grid)

        # get the total number of missing points in each sample:
        nb_missing_points = torch.isnan(series_c).sum(1).squeeze() # (bs)
        if nb_missing_points.ndim == 0:
            nb_missing_points = nb_missing_points.unsqueeze(0)
        
        # compute the maximum context length:
        max_context_len   = int( series_c.shape[1] - min(nb_missing_points) )

        # prepare context tensors; handle each sample separately:
        list_coords_c, list_series_c, list_covar_c = [], [], []
        for idx in range(len(series_c)):
            
            # get sample:
            sample = series_c[idx].squeeze()

            # get missing mask
            is_missing_mask = torch.isnan(sample)                        # (T,)

            # handle series that have too many nans (= more than the min nb of nan in the batch)
            if is_missing_mask.sum() > min(nb_missing_points):

                # get number of values actually observed:
                cur_len = len(sample[~is_missing_mask])
                mis_len = max_context_len - cur_len

                # (i) pad context values with nans, up to length=max_context_len:
                # (nan values will be discared when we complete_nans)
                series_i = torch.cat([
                    sample[~is_missing_mask],
                    torch.nan * torch.ones(mis_len).to(sample.device)
                ]).unsqueeze(-1) # (max_context_len, 1)

                # (ii) same padding (with ones) for the grid:
                grid_i = torch.cat([
                    grid[idx][~is_missing_mask].squeeze(),
                    torch.ones(mis_len).to(sample.device) # (T, 1), (T, 1)
                ]).unsqueeze(-1) # (max_context_len, 1)

                # (iii) handle covariates (context part):
                if has_covar:

                    # check time alignment:
                    assert covariates.shape[-2] == series_c.shape[-2]
                    
                    # we keep only the covariates observed when the target series is observed
                    # + then we must pad with nans:
                    nb_covars = covariates.shape[1] # (bs C t 1)
                    covar_c = torch.cat([
                        covariates[idx][...,~is_missing_mask,:], # (c t_obs 1)
                        torch.nan * torch.ones((nb_covars, mis_len, 1)).to(sample.device)
                    ], dim = - 2) # (C, max_context_len, 1)

                    # now, each covar may also contain NaN (and they may not be aligned between themselves)

                # (iv) complete nans:
                out = self.complete_nans(series_i, grid_i, is_test=True, X_cov=covar_c)
                series_i, grid_i, covar_c = out['values'], out['coords'], out.get('covar')

            else:
                series_i, grid_i = series_c[idx][~is_missing_mask], grid[idx][~is_missing_mask] # (max_context_len, 1), (max_context_len, 1)
                if has_covar:
                    covar_c = covariates[idx][...,~is_missing_mask,:] # (C, max_context_len, 1)
                    assert covar_c.shape[-2] == series_i.shape[-2]
            
            # check that there is no nan in the context values:
            assert not torch.isnan(series_i).any()

            list_coords_c.append(grid_i.reshape(-1,1))   # (max_context_len, 1)
            list_series_c.append(series_i.reshape(-1,1)) # (max_context_len, 1)
            if covariates is not None:
                list_covar_c.append(covar_c)             # (C, max_context_len, 1)

        coords_c = torch.stack(list_coords_c) # (bs, max_context_len, 1)
        series_c = torch.stack(list_series_c) # (bs, max_context_len, 1)

        # XXX query the entire grid (context and missing points):
        coords_t = grid # (bs, seq_len, 1)

        if has_covar:
            covariates_c = torch.stack(list_covar_c)    # (bs, C, max_context_len, 1)

        out = {
            'series_c' : series_c,
            'coords_c' : coords_c,
            'coords_t' : coords_t,
        }

        if has_covar:
            out['covar_c'] = covariates_c # (bs, C, max_context_len, 1)
            out['covar_t'] = covariates   # (bs, C, query_seq_len, 1)

        return out

        # return series_c, coords_c, coords_t

    def get_context_target(
        self,
        grid: torch.Tensor,
        series_c: torch.Tensor,
        covariates: torch.Tensor | None = None,
        horizon_len: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract context and target tensors of the forecasting or imputation tasks.

        Args:
            grid (torch.Tensor): raw time grid of shape `(bs, T_grid, 1)`
            series_c (torch.Tensor): tensor of past values of shape `(bs, T, 1)`
            horizon_len (int): prediction horizon (forecasting only)
        
        Returns:
            Values on the context, time coords on the context and the query points.
        """

        assert grid.ndim == 3        
        assert series_c.ndim == 3
        assert series_c.shape[-1] == 1

        if self.do_forecasting:
            assert isinstance(horizon_len, int)
            return self._get_context_target_coords_f(grid = grid, series_c = series_c, horizon_len = horizon_len, covariates = covariates)
        else:
            return self._get_context_target_coords_i(grid = grid, series_c = series_c, covariates = covariates)

    # ---------------
    # MAIN UI
    # ---------------

    # XXX TODO add covar option to rollout forecast

    def predict_quantiles(
        self,
        inputs,
        covars: List[torch.Tensor] | torch.Tensor | None = None,
        prediction_length: int = 0,
        batch_size: int = 64,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        device: torch.device = torch.device('cuda'),
        denormalize: bool = False,
        freq: Optional[str] = None
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Inference pass of TS-ICL.

        Args:
            inputs: available context values either has a tensor or a list of tensors
            prediction_length (int): queried horizon length (forecasting only)
            batch_size (int): max batch size to process the inputs
            quantiles_levels (list[float]): list of quantiles to fetch
            device (torch.device): device to map the model to
            denormalize (bool): whether the undo z-normalization after predicting
            freq (str): *not used*
        
        Returns:
            Forecasts or imputed values as a tensor of shape `(bs, seq_len, C, 1)`.
        """

        # 0/3 INITIALIZE PIPELINE

        self.eval()
        self.to(device)

        # get quantile indices:
        quantile_levels = [100 * x for x in quantile_levels]
        assert set(quantile_levels).issubset(self.training_quantile_levels)
        quantile_indices = [self.training_quantile_levels.index(q) for q in quantile_levels]
        
        # define grid length:
        grid_len = self._get_grid_len()


        # 1/3 INPUT ADAPTER

        # prepare inputs:
        inputs, covars, num_batches, is_tensor, has_covar = self.input_adapter(
            inputs     = inputs,
            covars     = covars,
            batch_size = batch_size
        )
        # inputs is now a tensor (bs, t, 1) or a list of tensors (1, t, 1)


        # 2/3 RUN BATCH-WISE INFERENCE

        # allocate some space:
        all_forecasts = []

        # main loop:
        for idx in range(num_batches):

            # 
            series_c = inputs[idx*batch_size:(idx+1)*batch_size] if is_tensor else inputs[idx]
            assert isinstance(series_c, torch.Tensor)
            series_c = series_c.to(device)

            if covars is not None:
                covars_idx = covars[idx*batch_size:(idx+1)*batch_size] if is_tensor else covars[idx]
                assert isinstance(covars_idx, torch.Tensor)
                covars_idx = covars_idx.to(device)
            else:
                covars_idx = None

            # build context and target grids:
            grid = self.make_grid(grid_len, num_samples=len(series_c)).to(device)

            if prediction_length > self.max_target_length and self.do_forecasting:
                
                # XXX does not support covars
                # series_c (bs, T, 1)

                n_rollouts = prediction_length // self.max_target_length + int(prediction_length % self.max_target_length > 0)
                yhat = []

                for ii in range(n_rollouts):

                    # get prediction length:
                    pred_len = prediction_length % self.max_target_length if ii == (n_rollouts - 1) else self.max_target_length

                    # prepare tensors:
                    processed_inputs = self.get_context_target(
                        grid        = grid,
                        horizon_len = pred_len,
                        series_c    = series_c
                    )
                    series_c = processed_inputs['series_c']
                    coords_c = processed_inputs['coords_c']
                    coords_t = processed_inputs['coords_t']

                    # predict:
                    # (znorm -> asinh -> pred --> sinh -> znorm^-1)
                    yhat_ii = self._predict_batch(
                        series_c, coords_c, coords_t, denormalize=True, save_scaler=(ii==0)
                    )
                    yhat.append(yhat_ii)

                    # update context:
                    series_c = torch.cat([
                        series_c,                          # (bs, t_in, 1)
                        yhat_ii.mean(dim=-1, keepdim=True) # (bs, t_mis, 1)
                    ], dim = 1)
                
                # stack forecasts: 
                yhat = torch.cat(yhat, dim=1) # (bs, t, q)
                if not denormalize:
                    yhat = self.scaler.transform(yhat)

            else:
                processed_inputs = self.get_context_target(
                    grid        = grid,
                    horizon_len = prediction_length,
                    series_c    = series_c,
                    covariates  = covars_idx
                )
                series_c = processed_inputs['series_c'] # (bs, max_context_len, 1)
                coords_c = processed_inputs['coords_c'] # (bs, max_context_len, 1)
                coords_t = processed_inputs['coords_t'] # (bs, query_seq_len, 1)

                if has_covar:
                    covar_c = processed_inputs['covar_c']
                    covar_t = processed_inputs.get('covar_t')
                else:
                    covar_c, covar_t = None, None

                # predict:
                yhat = self._predict_batch(series_c, coords_c, coords_t, covar_c=covar_c, covar_t=covar_t, denormalize=denormalize)

            all_forecasts.append(yhat[...,quantile_indices].cpu().detach())


        # 3/3 COLLECT FORECASTS

        # stack all preds:
        if isinstance(inputs, list):
            all_forecasts = [rearrange(x, 'c t q -> t c q') for x in all_forecasts]
        else:
            all_forecasts = torch.cat(all_forecasts, 0)
            all_forecasts = rearrange(all_forecasts, '(b c) t q -> b t c q', c=self.num_var)

        # check dims:
        if is_tensor:
            inputs = rearrange(inputs, '(b c) t 1 -> b t c 1', c=self.num_var)
        assert len(all_forecasts) == len(inputs)
        
        return all_forecasts

    # ---------------
    # FORWARD PASS
    # ---------------

    def _predict_batch(
        self,
        series_c: torch.Tensor,
        coords_c: torch.Tensor,
        coords_t: torch.Tensor,
        covar_c: torch.Tensor | None = None,
        covar_t: torch.Tensor | None = None,
        denormalize: bool = False,
        save_scaler: bool = False
    ) -> torch.Tensor:
        """ 
        Run one forward of TS-ICL on batched data.

        Args:
            series_c (torch.Tensor): `(bs, T_ctx, 1)` tensor of observed values
            coords_c (torch.Tensor): `(bs, T_ctx, 1)` tensor of observed coords
            coords_t (torch.Tensor): `(bs, T_mis, 1)` tensor of queried coords
            covar (torch.Tensor, optional): `(bs, C, T_ctx + T_mis, 1)` tensor of covariates
            denormalize (bool): whether the undo z-normalization after predicting
        
        Returns:
            Predicted target quantiles of shape `(bs, T_mis, Q)`.
        """

        series_covar = None

        # replace nans:
        if self.do_forecasting:
            out = self.complete_nans(series_c, coords_c, is_test=True)
            series_c, coords_c = out['values'], out['coords']

        # z-normalize context series:
        scaler   = CustomStandardScaler(dim=1,epsilon=1e-5)
        scaler.fit(series_c)
        series_c_norm = scaler.transform(series_c)

        # z-normalize covariate series:
        if (covar_c is not None) and (covar_t is not None):
            scaler_cov = CustomStandardScaler(dim=-2,epsilon=1e-7)
            scaler_cov.fit(covar_t) # type: ignore
            covar_t      = scaler_cov.transform(covar_t)         # (bs, C, query_seq_len, 1)
            covar_c      = scaler_cov.transform(covar_c)         # (bs, C, max_context_len, 1)
            series_covar = torch.cat([covar_c, covar_t], dim=-2) # (bs, C, max_context_len + query_seq_len, 1)

        # build query coords:
        query_coords = torch.cat([coords_c, coords_t], dim=1) # [N, T_in+T, 1]

        # get predictions:
        quantiles, _ = self(
            series               = series_c_norm,
            coords               = coords_c,
            target_coords        = query_coords,
            series_covar         = series_covar,
            coords_covar         = query_coords,
            undo_asinh_transform = True
        )

        if save_scaler:
            self.scaler = scaler

        if denormalize:
            quantiles = scaler.inv_transform(quantiles)
            # if std_mask.sum() > 0:
            #     quantiles[std_mask] = repeat(series_c[std_mask].mean(dim=1,keepdim=True), 'b 1 q -> b h q', h = quantiles.shape[1])
    
        return quantiles
    