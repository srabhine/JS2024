from typing import Optional, Union, List, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_feature_by_symbols(data_ixd: pd.DataFrame,
                            name_feat: str,
                            syms: Optional[Union[int, List[int]]] =
                            None,
                            figs_dir: Optional[Any] = None):
    if syms is None:
        syms = list(np.arange(39))

    n_syms = len(syms)
    n_rows, n_cols = square_grid(n_syms)
    k = 0
    f, axs = plt.subplots(nrows=n_rows,
                          ncols=n_cols, figsize=(20, 20))
    f.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(n_rows):
        for j in range(n_cols):
            if k < n_syms:
                sym = syms[k]
                df_tmp = data_ixd.loc[sym]
                axs[i, j].plot(df_tmp[name_feat].values)
                axs[i, j].set(title=sym)
            else:
                plt.delaxes(axs[i, j])
            k += 1
    if figs_dir is not None:
        f.savefig(figs_dir / f'{name_feat}_by_symbol.png')
        plt.close(f)

    return f, axs
