import sys
import os
import subprocess
import socket
import getpass
import time
import json
from pathlib import Path
from typing import Dict, Any

def get_git_revision() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        return "unknown"

def get_base_metadata() -> Dict[str, Any]:
    return {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'hostname': socket.gethostname(),
        'user': getpass.getuser(),
        'python_version': sys.version.split()[0],
        'git_revision': get_git_revision(),
        'cwd': os.getcwd(),
        'pid': os.getpid()
    }

def log_run(run_dir: Path, config: Dict, extra_meta: Dict[str, Any]):
    """
    Writes run.json with strict validation.
    Required extra_meta keys:
    - dt, dt_eff
    - N, trials, trial_T
    - tau_list, seed_list
    - beta_E_list, beta_C_list
    """
    required_keys = [
        'dt', 'dt_eff', 
        'N', 'trials', 'trial_T', 
        'tau_list', 'seed_list', 
        'beta_E_list', 'beta_C_list'
    ]
    
    missing = [k for k in required_keys if k not in extra_meta]
    if missing:
        print(f"CRITICAL ERROR: Missing required run metadata keys: {missing}")
        print("Run validation failed. Exiting.")
        sys.exit(1)
        
    full_log = get_base_metadata()
    full_log['config'] = config
    full_log.update(extra_meta)
    
    with open(run_dir / "run.json", "w") as f:
        json.dump(full_log, f, indent=2, default=str)
    print(f"Run metadata verified and written to {run_dir / 'run.json'}")
