import numpy as np


def convolve_spikes(spikes: np.ndarray, tau: float, dt: float, warmup: float = 0.0) -> np.ndarray:
    """Low-pass filter spike trains with a causal exponential kernel.

    Parameters
    ----------
    spikes:
        Either
          - shape (T, N): single trial, N neurons, 0/1 spike indicators per bin
          - shape (n_trials, T, N): batched trials
    tau:
        Exponential filter time constant (seconds). If tau <= 0, the raw firing
        rate is returned.
    dt:
        Time step (seconds).
    warmup:
        Duration (seconds) to discard from the *start* of the returned filtered
        trace(s).

    Returns
    -------
    A_tau:
        Filtered population rate in units of spikes/s per neuron.
        Shape is (T',) for 2D input and (n_trials, T') for 3D input.

    Notes
    -----
    Implements the causal ODE filter
        dA/dt = (r(t) - A)/tau,
    via forward Euler:
        A_{t+1} = A_t + (dt/tau) (r_t - A_t).
    """
    spikes = np.asarray(spikes)

    # Normalize input shapes
    if spikes.ndim == 2:
        spikes_ = spikes[None, ...]  # (1, T, N)
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

    # Raw per-neuron firing rate (spikes/s)
    r = spikes_.mean(axis=-1) / float(dt)

    if tau is None or tau <= 0:
        A = r
    else:
        alpha = float(dt) / float(tau)
        # Clamp for stability if tau < dt
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
