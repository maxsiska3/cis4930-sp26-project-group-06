"""Run E2 with bounded native-library threading to avoid CPU oversubscription.

Usage: python run_e2.py --raw --out results/e2_raw.csv
All experiment arguments are passed to e2_rare_family unchanged.
"""
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

from threadpoolctl import threadpool_limits
from e2_rare_family import main

if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
