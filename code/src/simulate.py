import numpy as np
from src.stimuli import get_stimulus
from src.model import PopulationModel

def run_simulation(cfg):
    seed = cfg['simulation'].get('seed', 42)
    np.random.seed(seed)
    
    T = cfg['simulation'].get('T', 10.0)
    dt = cfg['simulation'].get('dt', 0.001)
    
    stim_cfg = {'dt': dt, 'seed': seed, **cfg.get('stimulus', {})} 
    
    stim_model = get_stimulus(cfg['stimulus']) 
    S = stim_model.generate(T, dt)
    
    model = PopulationModel(cfg)
    
    
    A, spikes = model.run_batch(S, dt)
    
    burn_in = cfg['simulation'].get('burn_in', 0.0)
    n_burn = int(burn_in / dt)
    
    S_valid = S[n_burn:]
    A_valid = A[n_burn:]
    spikes_valid = spikes[n_burn:]
    
    mean_rate = np.mean(A_valid)
    energy = {'mean_rate': mean_rate, 'total_spikes': mean_rate * len(A_valid) * dt * model.N}

    return {
        'S': S_valid,
        'A': A_valid,
        'spikes': spikes_valid,
        'dt': dt,
        'energy': energy,
        'metadata': {
            'T': T, 
            'N': model.N, 
            'theta0': model.theta0
        }
    }
