import numpy as np
from scipy import signal

def estimate_mi_upper_gaussian(S, A, dt, cfg):
   
    
    nperseg = int(cfg.get('window_size', 0.1) / dt)
    nperseg = max(256, nperseg)
    nperseg = min(len(S), nperseg)
    
    fs = 1.0 / dt
    f, Cxy = signal.coherence(S, A, fs=fs, nperseg=nperseg)
    
   
    Cxy = np.clip(Cxy, 0.0, 0.99999)
    
    integrand = -0.5 * np.log2(1.0 - Cxy) 
    
    
    
    mi_rate = np.trapz(integrand, f) 
    
    return {
        'I_upper_surrogate_bits_per_s': mi_rate, 
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
