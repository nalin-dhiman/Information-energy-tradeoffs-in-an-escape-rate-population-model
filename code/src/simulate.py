import numpy as np
from src.stimuli import get_stimulus
from src.model import PopulationModel

def run_simulation(cfg):
    # Setup
    seed = cfg['simulation'].get('seed', 42)
    np.random.seed(seed)
    
    T = cfg['simulation'].get('T', 10.0)
    dt = cfg['simulation'].get('dt', 0.001)
    
    # Stimulus
    stim_cfg = {'dt': dt, 'seed': seed, **cfg.get('stimulus', {})} # allow inline
    # If cfg['stimulus'] is a path string, it should have been loaded by the config loader already?
    # No, base.yaml has "stimulus: configs/stimulus/..." path.
    # The config loader I wrote in src/io/config.py does NOT auto-load sub-configs referenced by path.
    # I need to handle that. 
    # BUT, the `run_simulation` usually receives the fully resolved config dict?
    # Or I should resolve it here.
    # For now, let's assume the passed cfg has the resolved parameters OR the caller (script) handles loading.
    # The script `b0_sanity...` will load base, then load stimulus config and merge?
    # Or `run_simulation` does it.
    # Better: `run_simulation` assumes `cfg` is fully populated.
    
    # Generate Stimulus
    # We need to instantiate the Stimulus class.
    # But `get_stimulus` expects a dict.
    # If `cfg['stimulus']` is just a string path, we have a problem.
    # The CLI loader I wrote is simple.
    # I should update `config.py` to recursively load or do it in the script.
    # I'll assume standard practice: the script loads the stimulus config and puts it into `cfg['stimulus']`.
    
    stim_model = get_stimulus(cfg['stimulus']) # Expecting dict
    S = stim_model.generate(T, dt)
    
    # Model
    model = PopulationModel(cfg)
    
    # Run
    # We need a more efficient runner than the one inside PopulationModel if we want to save S, A, etc.
    # Actually, PopulationModel.run_batch is what I wrote.
    # Let's use it or rewrite logic here?
    # The prompt says "Implement src/simulate.py... run_simulation(cfg) -> dict".
    # I'll let the model handle the stepping.
    
    # But I see I implemented `run_batch` in `model.py` to return `pop_rate`.
    # I should update `model.py` to be more flexible or just use it.
    
    # Returns (pop_rate, spikes)
    A, spikes = model.run_batch(S, dt)
    
    # Burn in
    burn_in = cfg['simulation'].get('burn_in', 0.0)
    n_burn = int(burn_in / dt)
    
    S_valid = S[n_burn:]
    A_valid = A[n_burn:]
    spikes_valid = spikes[n_burn:]
    # Spikes need to be captured from model.run_batch?
    # model.run_batch returns pop_rate (A).
    # We need to modify model.run_batch to return spikes too.
    # OR, we modify model.run_batch to return (A, spikes).
    # See next tool call.
    
    # Energy Proxy
    # E ~ mean rate
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
