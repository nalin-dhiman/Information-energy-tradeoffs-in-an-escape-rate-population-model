import numpy as np


def entropy_gaussian(x: np.ndarray, eps: float = 1e-12) -> float:
   
    x = np.asarray(x).ravel()
    if x.size == 0:
        return float("nan")

    var = float(np.var(x, ddof=1)) if x.size > 1 else 0.0
    var = max(var, eps)
    return 0.5 * np.log2(2.0 * np.pi * np.e * var)
