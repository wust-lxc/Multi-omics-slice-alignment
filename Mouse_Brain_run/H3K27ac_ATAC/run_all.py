#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path


def configure_runtime_threads() -> None:
    target_threads = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for key, value in target_threads.items():
        os.environ[key] = value


def run_step(script_path: Path) -> None:
    print(f"[RUN] {script_path}")
    subprocess.run([sys.executable, str(script_path)], check=True)


def main() -> int:
    configure_runtime_threads()

    run_dir = Path(__file__).resolve().parent
    steps = [
        run_dir / "01_prepare_data.py",
        run_dir / "02_embedding_alignment.py",
        run_dir / "03_slice_order_and_z_reconstruction.py",
        run_dir / "04_location_alignment.py",
        run_dir / "05_chamfer_distance.py",
    ]

    for step in steps:
        if not step.exists():
            raise FileNotFoundError(f"Missing pipeline step: {step}")

    try:
        for step in steps:
            run_step(step)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Pipeline failed at: {exc.cmd}")
        return exc.returncode

    print("Mouse Brain H3K27ac-ATAC pipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
