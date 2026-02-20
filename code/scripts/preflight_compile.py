import compileall
import sys
from pathlib import Path

def check_syntax(src_dir):
    print(f"Compiling source in {src_dir}...")
    if not compileall.compile_dir(src_dir, force=True, quiet=1):
        print("Syntax check FAILED.")
        sys.exit(1)
    print("Syntax check PASSED.")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.resolve()
    src_dir = base_dir / "src"
    check_syntax(src_dir)
