import numpy as np


def entropy_gaussian(x: np.ndarray, eps: float = 1e-12) -> float:
    """Differential entropy of a 1D Gaussian fit to samples.

    Computes
        h(X) = 0.5 * log2(2*pi*e*sigma^2)

    Parameters
    ----------
    x:
        Samples of the variable.
    eps:
        Floor added to the variance for numerical stability.

    Returns
    -------
    h:
        Differential entropy in **bits**.

    Notes
    -----
    This is a model-based entropy estimate (Gaussian assumption). For strongly
    non-Gaussian distributions, use a nonparametric estimator instead.
    """
    x = np.asarray(x).ravel()
    if x.size == 0:
        return float("nan")

    var = float(np.var(x, ddof=1)) if x.size > 1 else 0.0
    var = max(var, eps)
    return 0.5 * np.log2(2.0 * np.pi * np.e * var)
