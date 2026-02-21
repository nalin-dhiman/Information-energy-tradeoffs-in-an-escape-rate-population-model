import string
from pathlib import Path
from typing import Optional, List


def get_next_run_id(runs_dir: Path, major_version: int) -> str:
   
    base_id = f"v{major_version}"
    if not (runs_dir / base_id).exists():
        return base_id

    for suffix in string.ascii_lowercase:
        run_id = f"{base_id}_{suffix}"
        if not (runs_dir / run_id).exists():
            return run_id

    raise RuntimeError(
        f"Too many runs for major version {major_version}. "
        "Please increment the major version."
    )


def create_run_dir(major_version: int, root: Optional[Path] = None) -> Path:
    
    root = Path(root) if root is not None else Path.cwd()
    runs_root = root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    run_id = get_next_run_id(runs_root, major_version)
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    for sub in ("figures", "tables", "logs"):
        (run_dir / sub).mkdir(exist_ok=True)

    (runs_root / "LATEST").write_text(run_id + "\n")

    return run_dir


def _list_run_dirs(runs_root: Path) -> List[Path]:
    if not runs_root.exists():
        return []
    dirs = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("v")]
    return sorted(dirs, key=lambda p: p.stat().st_mtime)


def get_latest_run_dir(root: Optional[Path] = None) -> Path:
  
    root = Path(root) if root is not None else Path.cwd()
    runs_root = root / "runs"

    latest_file = runs_root / "LATEST"
    if latest_file.exists():
        token = latest_file.read_text().strip()
        if token:
            candidate = runs_root / token
            if candidate.exists():
                return candidate

            abs_candidate = Path(token)
            if abs_candidate.is_absolute() and abs_candidate.exists():
                return abs_candidate

    run_dirs = _list_run_dirs(runs_root)
    if run_dirs:
        return run_dirs[-1]

    raise FileNotFoundError(
        f"No run directories found under {runs_root}. "
        "Run a sweep first (e.g., scripts/b0_sanity_sweep_theta0.py)."
    )
