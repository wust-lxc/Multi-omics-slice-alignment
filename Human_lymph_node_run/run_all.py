#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path


def run_step(script_path: Path) -> None:
    print(f"[RUN] {script_path}")
    subprocess.run([sys.executable, str(script_path)], check=True)


def configure_runtime_threads() -> None:
    target_threads = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for key, target_value in target_threads.items():
        current = os.environ.get(key)
        if current != target_value:
            os.environ[key] = target_value


def main() -> int:
    configure_runtime_threads()

    repo_root = Path(__file__).resolve().parent.parent
    run_dir = repo_root / "Human_lymph_node_run"

    steps = [
        run_dir / "01_prepare_data.py",
        run_dir / "02_embedding_alignment.py",
        run_dir / "03_slice_order_and_z_reconstruction.py",
        run_dir / "04_location_alignment.py",
        run_dir / "05_build_3d_and_export.py",
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

    print("Human lymph node pipeline finished. Outputs are under ./Human_lymph_node_result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
