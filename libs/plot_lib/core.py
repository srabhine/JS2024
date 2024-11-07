import numpy as np


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
