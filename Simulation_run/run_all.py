#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path


def _python_has_scanpy(python_exe: str) -> bool:
    try:
        subprocess.run(
            [python_exe, "-c", "import scanpy"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def resolve_python() -> str:
    if _python_has_scanpy(sys.executable):
        return sys.executable

    stair_env_python = Path("/root/miniconda3/envs/STAIR-env/bin/python")
    if stair_env_python.exists() and _python_has_scanpy(str(stair_env_python)):
        print(f"[INFO] Current Python lacks scanpy; using {stair_env_python}")
        return str(stair_env_python)

    return sys.executable


def run_step(script_path: Path, python_exe: str) -> None:
    print(f"[RUN] {script_path}")
    subprocess.run([python_exe, str(script_path)], check=True)


def configure_runtime_threads(python_exe: str) -> None:
    target_threads = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for key, target_value in target_threads.items():
        if os.environ.get(key) != target_value:
            os.environ[key] = target_value

    if Path("/usr/bin/Rscript").exists():
        os.environ["R_HOME"] = "/usr/lib/R"
        os.environ["PATH"] = "/usr/bin" + os.pathsep + os.environ.get("PATH", "")
        return

    env_prefix = str(Path(python_exe).resolve().parent.parent)
    os.environ["CONDA_PREFIX"] = env_prefix
    os.environ["R_HOME"] = str(Path(env_prefix) / "lib" / "R")
    os.environ["PATH"] = str(Path(env_prefix) / "bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = str(Path(env_prefix) / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")


def main() -> int:
    python_exe = resolve_python()
    configure_runtime_threads(python_exe)

    repo_root = Path(__file__).resolve().parent.parent
    run_dir = repo_root / "Simulation_run"

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
            run_step(step, python_exe)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Pipeline failed at: {exc.cmd}")
        return exc.returncode

    print("Simulation pipeline finished. Outputs are under ./Simulation_result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
