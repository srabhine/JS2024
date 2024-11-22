import numpy as np


PARAMS_HELPER = dict(ls='-.', c='k', lw=0.5, label='')
PARAMS_HELPER_SANS_CLR = dict(ls='-.', lw=0.5, label='')
PARAMS_HELPER_SANS_LBL = dict(ls='-.', lw=0.5)
PARAMS_HELPER_LBL = dict(ls='-.', c='k', lw=0.5, label='helper')
PARAMS_HELPER_SANS_CLR_LBL = dict(ls='-.', lw=0.5)


def square_grid(n: int):
    """
    Given a number of n subplots it returns the 'best' number of
    rows and columns to distribute n subplots in a matrix of subplots

    Parameters
    ----------
    n: int
        Number of subplots

    Returns
    -------

    """
    n_rows = int(np.round(np.sqrt(n)))
    n_cols = 1 if n == 1 else n // n_rows + 1

    return n_rows, n_cols
