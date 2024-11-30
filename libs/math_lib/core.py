"""

@author: Raffaele Ghigliazza
"""

from typing import Optional, Any
import numpy as np


def r2_zero(y_true, y_pred, weights: Optional[Any] = None):
    if weights is None:
        weights = 1 / len(y_true) * np.ones_like(y_pred)
    num = (weights * (y_true - y_pred)**2).sum()
    den = (weights * y_true**2).sum()
    return 1 - num / den

