"""
Temporary script: clean all __pycache__ directories and .pyc files recursively.
Run from project root: python clean_cache.py
"""
import os
from pathlib import Path

def clean_pycache(root: Path) -> tuple[int, int]:
    removed_dirs = 0
    removed_files = 0
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        # Remove .pyc files
        for name in filenames:
            if name.endswith(".pyc"):
                p = Path(dirpath) / name
                try:
                    p.unlink()
                    removed_files += 1
                except OSError:
                    pass
        # Remove __pycache__ directories
        for name in dirnames:
            if name == "__pycache__":
                p = Path(dirpath) / name
                try:
                    p.rmdir()
                    removed_dirs += 1
                except OSError:
                    pass
    return removed_dirs, removed_files

def main():
    root = Path(__file__).resolve().parent
    print(f"Cleaning bytecode cache under: {root}")
    dirs, files = clean_pycache(root)
    print(f"Removed {dirs} __pycache__ folder(s) and {files} .pyc file(s).")
    print("Done. Restart the backend to use fresh code.")

if __name__ == "__main__":
    main()
