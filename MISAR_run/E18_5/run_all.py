from pathlib import Path

from pipeline import run_all


if __name__ == "__main__":
    raise SystemExit(run_all(Path(__file__).resolve().parent))
