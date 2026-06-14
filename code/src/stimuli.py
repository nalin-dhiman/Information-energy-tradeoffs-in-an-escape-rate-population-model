import numpy as np
from scipy.ndimage import gaussian_filter1d

class StimulusBase:
    def generate(self, T, dt):
        raise NotImplementedError

class GaussianBandLimited(StimulusBase):
    def __init__(self, cfg):
        self.cutoff_hz = cfg.get('cutoff_hz', cfg.get('cutoff_freq', 20.0))
        if 'tau_c' in cfg:
             # Map tau_c (correlation time) to cutoff freq
             # f_c = 1 / (2 * pi * tau_c)
             self.cutoff_hz = 1.0 / (2.0 * np.pi * cfg['tau_c'])
        self.mean = cfg.get('mean', 0.0)
        self.std = cfg.get('std', 1.0)
        self.seed = cfg.get('seed', None)
        
    def generate(self, T, dt):
        if self.seed is not None:
            np.random.seed(self.seed)
        n_steps = int(T / dt)
        white = np.random.randn(n_steps)
        # Smooth to bandlimit
        sigma = 1.0 / (2 * np.pi * self.cutoff_hz * dt)
        filtered = gaussian_filter1d(white, sigma)
        # Normalize
        filtered = (filtered - np.mean(filtered)) / np.std(filtered)
        return self.mean + self.std * filtered

class GaussSwitching(StimulusBase):
    def __init__(self, cfg):
        self.mean = cfg.get('mean', 0.0)
        self.std_low = cfg.get('std_low', 0.5)
        self.std_high = cfg.get('std_high', 2.0)
        self.rate = cfg.get('switch_rate', 0.1)
        if 'tau_c' in cfg:
             # Telegraph process correlation time tau = 1 / (2 * rate)
             # So rate = 1 / (2 * tau)
             self.rate = 1.0 / (2.0 * cfg['tau_c'])
        self.std = cfg.get('std', 1.0) # Target global std
        self.seed = cfg.get('seed', 42)
        
    def generate(self, T, dt):
        np.random.seed(self.seed)
        n_steps = int(T / dt)
        # Generate state (0 or 1)
        # Poisson switching
        switch_prob = self.rate * dt
        state = 0
        states = np.zeros(n_steps)
        for i in range(n_steps):
            if np.random.rand() < switch_prob:
                state = 1 - state
            states[i] = state
            
        noise = np.random.randn(n_steps)
        stds = np.where(states == 0, self.std_low, self.std_high)
        raw_signal = self.mean + stds * noise
        
        # Enforce target variance (standardization)
        # We want the final signal to have std = self.target_std (default 1.0)
        # But GaussSwitching has specific structure. 
        # If we normalize, we maintain the relative ratio of low/high variance but fix global power.
        target_std = self.std # Use the 'std' param as target (defaults/inherited)
        # Note: 'std' key might not be in cfg if not set, let's use a default or self.std if set.
        # But __init__ didn't set self.std. Let's fix __init__ to read 'std'.
        
        # normalizing
        current_std = np.std(raw_signal)
        if current_std < 1e-9: current_std = 1.0
        normalized = (raw_signal - np.mean(raw_signal)) / current_std
        return self.mean + self.std * normalized

class OneOverF(StimulusBase):
    def __init__(self, cfg):
        self.exponent = cfg.get('exponent', 1.0)
        self.f_min = cfg.get('f_min', 0.1)
        self.f_max = cfg.get('f_max', 100.0)
        self.mean = cfg.get('mean', 0.0)
        self.std = cfg.get('std', 1.0)
        self.seed = cfg.get('seed', 42)

    def generate(self, T, dt):
        np.random.seed(self.seed)
        n_steps = int(T / dt)
        # Colored noise via FFT
        freqs = np.fft.rfftfreq(n_steps, d=dt)
        spectrum = np.zeros_like(freqs)
        mask = (freqs >= self.f_min) & (freqs <= self.f_max)
        spectrum[mask] = 1.0 / (freqs[mask] ** (self.exponent / 2.0))
        
        phases = np.random.rand(len(freqs)) * 2 * np.pi
        complex_spec = spectrum * np.exp(1j * phases)
        signal = np.fft.irfft(complex_spec, n=n_steps)
        
        # Normalize
        signal = (signal - np.mean(signal)) / np.std(signal)
        return self.mean + self.std * signal

def get_stimulus(cfg):
    type_map = {
        'gauss_bandlimited': GaussianBandLimited,
        'gauss_switching': GaussSwitching,
        'one_over_f': OneOverF
    }
    stim_type = cfg.get('type', 'gauss_bandlimited')
    if stim_type not in type_map:
        raise ValueError(f"Unknown stimulus type: {stim_type}")
    return type_map[stim_type](cfg)
