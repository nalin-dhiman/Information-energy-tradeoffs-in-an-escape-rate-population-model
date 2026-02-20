import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.io.paths import create_run_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--major', type=int, default=1, help='Major version number')
    args = parser.parse_args()
    
    run_dir = create_run_dir(major_version=args.major)
