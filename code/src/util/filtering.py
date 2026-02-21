import numpy as np


def convolve_spikes(spikes: np.ndarray, tau: float, dt: float, warmup: float = 0.0) -> np.ndarray:
    
    spikes = np.asarray(spikes)

    if spikes.ndim == 2:
        spikes_ = spikes[None, ...] 
        squeeze = True
    elif spikes.ndim == 3:
        spikes_ = spikes
        squeeze = False
    else:
        raise ValueError(
            f"spikes must have shape (T, N) or (n_trials, T, N); got {spikes.shape}"
        )

    n_trials, T, N = spikes_.shape
    if N <= 0:
        raise ValueError("N must be positive")

    r = spikes_.mean(axis=-1) / float(dt)

    if tau is None or tau <= 0:
        A = r
    else:
        alpha = float(dt) / float(tau)
        alpha = min(max(alpha, 0.0), 1.0)
        A = np.empty_like(r)
        A[:, 0] = r[:, 0]
        for t in range(1, T):
            A[:, t] = A[:, t - 1] + alpha * (r[:, t] - A[:, t - 1])

    if warmup and warmup > 0:
        iw = int(round(float(warmup) / float(dt)))
        iw = min(max(iw, 0), T)
        A = A[:, iw:]

    if squeeze:
        return A[0]
    return A
