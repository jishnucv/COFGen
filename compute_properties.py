"""
compute_properties.py — batch property computation entry point.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.property_labels import compute_and_attach

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",      type=str, required=True)
    p.add_argument("--n_jobs",    type=int, default=1)
    p.add_argument("--n_grid",    type=int, default=25)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    compute_and_attach(Path(args.data), geometric=True,
                       n_grid=args.n_grid, n_jobs=args.n_jobs,
                       overwrite=args.overwrite)

if __name__ == "__main__":
    main()
