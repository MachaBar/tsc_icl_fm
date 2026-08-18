import argparse
from typing import Sequence, Optional, Union
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gmean

pretty_names = {
    "mean_weighted_sum_quantile_loss" : "WQL",
    "MSE[mean]": "MSE",
    "RMSE[mean]": "RMSE",
    "NRMSE[mean]": "NRMSE",
    "MAE[0.5]": "MAE"
}

def make_table(
    output_path: Path,
    list_datasets: Optional[Union[str, Sequence[str]]] = None,
    list_settings: Optional[Sequence[str]] = None,
    mae_filename: str = 'inference/metrics/mae_mixture.csv',
    metric_name: str = 'MAE',
    print_tex_table: bool = False
) -> None:
    """
    Scan `output_path` to read the csv files from `inference_mixture.py` and export to an aggregated file.
    """

    if list_datasets is None:
        list_datasets = [f.stem for f in Path(output_path).iterdir() if f.is_dir() and f.stem != '.hydra']
    
    if (list_settings is None) or (list_settings == 'imputation'):
        list_settings = ['pointwise_missing_1', 'pointwise_missing_2', 'blocks_missing_1', 'blocks_missing_2']
    elif list_settings == 'forecasting':
        list_settings = ['forecasting_{}d_{}d'.format(l,h) for l in [14,28,42] for h in [1,2,4,7,14]]

    for metric_name in ['MAE[0.5]', 'MSE[mean]', 'mean_weighted_sum_quantile_loss']:
        
        results = defaultdict(list)
        list_dataset = []
        list_settting = []

        for dataset in list_datasets:

            for setting in list_settings:

                filename = output_path / dataset / setting / mae_filename

                if filename.exists():
                    list_dataset.append(dataset)
                    list_settting.append(setting)
                    df = pd.read_csv(filename, index_col = 0)
                    results['nb_chunks'].append(df['Chunks'].iloc[0] if 'Chunks' in df.columns else -1)
                    results['Time (s)'].append(df['Time (s)'].iloc[0] if 'Time (s)' in df.columns else -1)
                    for key in df.index:
                        if ~df.loc[key].isna().any():
                            col_key = metric_name if metric_name in df.loc[key].keys() else '{} on Missing Values (norm)'.format(metric_name)
                            results[key].append(df.loc[key][col_key])
                    
        if len(list_dataset) == 0:
            return
        
        df = pd.concat([pd.DataFrame({'Dataset': list_dataset, 'Setting': list_settting}), pd.DataFrame(results)], axis=1)
        df = df.sort_values(by = ['Dataset', 'Setting'], ascending=True, ignore_index=True, key=lambda col: col.str.lower())

        is_int_cols   = df.dtypes[(df.dtypes == 'int64')].index    
        is_float_cols = df.dtypes[(df.dtypes == 'float64')].index

        mean_scores  = df[is_float_cols].apply(np.mean, axis=0)
        gmean_scores = df[is_float_cols].apply(gmean, axis=0)

        df.loc['gmean', is_float_cols] = gmean_scores
        df.loc['mean', is_float_cols]  = mean_scores
        df.loc['mean', is_int_cols] = df[is_int_cols].apply(np.sum, axis=0)
    
        # mean_score = ['mean score', None, df.nb_chunks.sum()] + [
        #     df.iloc[:,i+3].mean() for i in range(len(df.columns)-3)
        # ]

        # df.loc[len(df)] = mean_score
        df.to_csv( output_path / '{}_inference.csv'.format(pretty_names.get(metric_name, metric_name).lower()) )

        # print latex table:
        if print_tex_table:
            print(
                pd.DataFrame(df).to_latex(
                    index=False,
                    float_format="{:.3f}".format,
                    column_format='c'*df.shape[1]
                )
            )

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path')
    parser.add_argument('-d', '--datasets', default=None)
    parser.add_argument('-s', '--settings', default=None)
    parser.add_argument('-f', '--filename', default='inference/metrics/mae_mixture.csv')
    parser.add_argument('-m', '--metric', default='MAE')


    args = parser.parse_args()
    output_path = Path( args.path )


    make_table( output_path, args.datasets, args.settings, args.filename, args.metric )
