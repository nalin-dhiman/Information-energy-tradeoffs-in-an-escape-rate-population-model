import numpy as np
from scipy import signal

def estimate_mi_upper_gaussian(S, A, dt, cfg):
    """
    Computes MI upper bound using Gaussian approximation via spectral coherence.
    I = -0.5 * integral( log(1 - C_xy^2(f)) ) df
    
    Args:
        S: Stimulus array
        A: Activity array
        dt: Time step
        cfg: Estimator config
    """
    # Smooth A?
    # Actually, coherence handles frequency dependence. 
    # But if A is spikes, we should smooth or just use coherence on binned/raw.
    # The prompt says "smooth A(t) with tau".
    # But "simulate.py" already returns A (which is smoothed rate from model).
    # So we use A directly.
    
    nperseg = int(cfg.get('window_size', 0.1) / dt)
    nperseg = max(256, nperseg)
    nperseg = min(len(S), nperseg)
    
    fs = 1.0 / dt
    f, Cxy = signal.coherence(S, A, fs=fs, nperseg=nperseg)
    
    # Integrate -0.5 log(1 - C^2)
    # Clip Cxy to avoid log(0)
    Cxy = np.clip(Cxy, 0.0, 0.99999)
    
    integrand = -0.5 * np.log2(1.0 - Cxy) # bits
    
    # Integrate over frequency band
    # Stimulus limited?
    # If stim is bandlimited, Cxy outside is noise.
    # But if S has no power, Cxy is undefined or 0.
    # We should integrate up to Nyquist or cutoff.
    # For now, integrate full spectrum.
    
    mi_rate = np.trapz(integrand, f) # bits/second
    
    # Total MI = rate * T? or just return rate.
    # "Output: I_upper (bits or bits/s)"
    # Prompt says "compute stable spectral/proxy bound".
    
    return {
        'I_upper_surrogate_bits_per_s': mi_rate, # Proxyl
        'mi_rate': mi_rate,
        'units': 'bits/s',
        'mi_total': mi_rate * (len(S) * dt),
        'diagnostics': {
            'max_coherence': np.max(Cxy),
            'mean_coherence': np.mean(Cxy),
            'estimator_name': 'mi_upper_gaussian_coherence',
            'normalization': '-0.5*int(log(1-C^2))',
            'assumptions': 'Gaussian rate proxy',
            'smooth_tau': cfg.get('parameters', {}).get('smooth_tau', 'unknown')
        }
    }
