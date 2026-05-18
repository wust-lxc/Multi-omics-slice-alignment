#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path


def configure_runtime_threads() -> None:
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"


def run_step(script_path: Path) -> None:
    print(f"[RUN] {script_path}")
    subprocess.run([sys.executable, str(script_path)], check=True)


def main() -> int:
    configure_runtime_threads()
    run_dir = Path(__file__).resolve().parent
    steps = [
        run_dir / "01_prepare_data.py",
        run_dir / "02_embedding_and_clustering.py",
        run_dir / "03_export_results.py",
    ]
    try:
        for step in steps:
            run_step(step)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Pipeline failed at: {exc.cmd}")
        return exc.returncode
    print("Mouse Brain ATAC single-slice pipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
