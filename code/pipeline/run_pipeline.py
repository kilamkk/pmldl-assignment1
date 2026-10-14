"""
Automated pipeline: runs all three stages and repeats every 5 minutes.

  Stage 1: data engineering   (code/datasets/data_engineering.py)
  Stage 2: model engineering  (code/models/train_model.py)
  Stage 3: deployment         (docker compose up -d --build)

Usage (from the project root, with the virtual environment activated):
    python code/pipeline/run_pipeline.py          # run every 5 minutes (stop with Ctrl+C)
    python code/pipeline/run_pipeline.py --once   # run the pipeline only one time
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "pipeline.log"
INTERVAL_SECONDS = 5 * 60  # the assignment asks for a run every 5 minutes

# Each step is (name, command). sys.executable is the python of the current venv.
STEPS = [
    ("Stage 1: data engineering", [sys.executable, "code/datasets/data_engineering.py"]),
    ("Stage 2: model engineering", [sys.executable, "code/models/train_model.py"]),
    (
        "Stage 3: deployment",
        ["docker", "compose", "-f", "code/deployment/docker-compose.yml", "up", "-d", "--build"],
    ),
]


def log(message):
    """Print the message and also append it to logs/pipeline.log"""
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(name, command):
    """Run one step. Returns True if it finished successfully."""
    log(f"START {name}")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)

    # save the output of the step to the log file (useful for debugging)
    output = (result.stdout + result.stderr).strip()
    if output:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(output + "\n")

    if result.returncode == 0:
        log(f"DONE  {name}")
        return True
    log(f"FAILED {name} (exit code {result.returncode}). See {LOG_FILE}")
    return False


def run_pipeline():
    """Run all stages one after another. If a stage fails, the next ones are skipped,
    so the previously deployed version of the app keeps working."""
    log("=== Pipeline started ===")
    for name, command in STEPS:
        if not run_step(name, command):
            log("=== Pipeline stopped because of an error ===")
            return False
    log("=== Pipeline finished successfully ===")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="run the pipeline one time and exit")
    args = parser.parse_args()

    if args.once:
        sys.exit(0 if run_pipeline() else 1)

    log(f"Scheduler started: pipeline runs every {INTERVAL_SECONDS // 60} minutes. Press Ctrl+C to stop.")
    try:
        while True:
            started = time.time()
            run_pipeline()
            # wait so that runs START every 5 minutes (if a run is longer, start the next one right away)
            wait = max(0, INTERVAL_SECONDS - (time.time() - started))
            log(f"Next run in {int(wait)} seconds")
            time.sleep(wait)
    except KeyboardInterrupt:
        log("Scheduler stopped by user")


if __name__ == "__main__":
    main()
