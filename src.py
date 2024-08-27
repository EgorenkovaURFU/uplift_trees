import pandas as pd
import numpy as np
import scipy as sp


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


