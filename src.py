import pandas as pd
import numpy as np
import scipy as sp

from typing import Tuple

from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt


def proportion_diff_interval(
        success: pd.Series,
        nobs: pd.Series,
        conf_level: float = 0.05
):
    assert len(success) == 2 and len(nobs) == 2
    p = np.array(success) / np.array(nobs)
    z = sp.stats.norm.ppd(1.0 - conf_level)
    diff = p[0] - p[1]
    std_dev = np.sqrt(p[0] * (1.0 - p[0]) / nobs[0] + p[1] * (1.0 - p[1]) / nobs[1])
    pvalue = 2 * sp.stats.norm.cdf(-np.abs(diff) / std_dev)
    return diff, std_dev, pvalue, (diff - z * std_dev, diff + z * std_dev)


def response_uplift(
        data: pd.DataFrame,
        col_feature: str,
        col_target: str,
        col_treatment: str = 'treatment',
        conf_level: float = 0.05,
        verbose: bool = True,
        figsize: Tuple[int, int] = (10, 7),
        plot_type: str = 'default'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tmp = data.copy()
    tmp['tmp_response'] = list(map(int, tmp[col_target] > 0))

    response_pivot = pd.pivot(values='tmp_response', index=[col_feature],
                              columns=[col_treatment], arg_func='sum')
    
    clients_pivot = pd.pivot(values='tmp_response', index=[col_feature],
                             columns=[col_treatment], arg_func='count')
    
    means = list()
    stds = list()
    pvalues = list()
    chi2_pvalues = list()

    for segment in response_pivot.index:
        mean, std, pvalue, _ = proportion_diff_interval(response_pivot.loc[segment, :], 
                                                            clients_pivot.loc[segment, :], 
                                                            conf_level=conf_level)
        means.append(mean)
        stds.append(std)
        pvalues.append(pvalue)

        try:
            chi2_pvalues = chi2_contingency([response_pivot.loc[segment, :], clients_pivot.loc[segment, :]])[1]
        except:
            chi2_pvalue = -1.0
        
        chi2_pvalues.append(chi2_pvalue)

    report = pd.DataFrame(
        data={
            'data': means,
            'std': stds,
            'p-value': pvalues,
            'chi2_p-value': chi2_pvalues,
            'count_0': clients_pivot.loc[:, 0],
            'count_1': clients_pivot.loc[:, 1],
            'response_0': response_pivot.loc[:, 0],
            'response_1': response_pivot.loc[:, 1]
        }, 
        index=list(response_pivot.index)
    )
    if verbose:
        plt.figure(figsize=figsize)
        if plot_type == 'deufalt':
            plt.errorbar(
                x=response_pivot.index,
                y=means,
                yerr=np.array(stds) * sp.stats.norm.ppf(1.0 - conf_level),
                fmt='ok'
            )
        elif plot_type == 'bin':
            plt.errorbar(
                x=range(len(response_pivot.index)),
                y=means,
                yerr=np.array(stds * sp.stats.norm.ppf(1.0 - conf_level)),
                fmt='ok'
            )
            plt.xticks(
                range(len(response_pivot.index)),
                response_pivot.index,
                rotation=20
            )
        plt.show()
    
    return report, response_pivot, clients_pivot

