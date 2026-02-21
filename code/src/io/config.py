import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            merge_configs(base[k], v)
        else:
            base[k] = v
    return base

def parse_cli_overrides(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Path to configuration file')
    parser.add_argument('--set', action='append', help='Override config values key=value', default=[])
    
    args, unknown = parser.parse_known_args(args_list)
    
    config = load_config(args.config)
    
    for override in args.set:
        key, value = override.split('=')
        
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            current = current.setdefault(k, {})
            
        try:
            val = eval(value)
        except:
            val = value
            
        current[keys[-1]] = val
        
    return config

def save_config(config: Dict[str, Any], path: Path):
    with open(path, 'w') as f:
        yaml.dump(config, f)
