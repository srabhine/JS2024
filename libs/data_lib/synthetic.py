"""

@author: Raffaele M Ghigliazza
"""
import numpy as np

def generate_ar1(n: int = 100, x0: float = 0.0,
                 rho=0.5,
                 seed: int = 1234):
    x = np.zeros(n)
    np.random.seed(seed)
    e = np.random.randn(n)
    x[0] = x0

    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]

    return x
