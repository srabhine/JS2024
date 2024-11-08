"""

@author: Raffaele M Ghigliazza
"""
from typing import Optional, Union, List

import numpy as np
import pandas as pd


def create_lags(ds: pd.Series,
                lags: Optional[Union[List[int], np.ndarray]] = None) \
        -> pd.DataFrame:
    """
    Create lags for a time series

    Parameters
    ----------
    ds: pd.Series
        Input series
    lags: List[int]
        List of lags

    Returns
    -------

    """
    df_lags = ds.to_frame().copy()
    lags = lags if lags is not None else [1]
    for lag in lags:
        df_lags[lag] = ds.shift(lag)
    df_lags.columns = ['orig'] + [f'lag_{lag}' for lag in lags]
    return df_lags


def create_leads(ds: pd.Series,
                 leads:
                 Optional[Union[List[int], np.ndarray]] = None) \
        -> pd.DataFrame:
    """
    Create leads for a time series

    Parameters
    ----------
    ds: pd.Series
        Input series
    leads: List[int]
        List of leads

    Returns
    -------

    """
    df_leads = ds.to_frame().copy()
    leads = leads if leads is not None else [1]
    for lead in leads:
        df_leads[lead] = ds.shift(-lead)
    df_leads.columns = ['orig'] + [f'lead_{lead}' for lead in leads]
    return df_leads
