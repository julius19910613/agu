#!/usr/bin/env python3
"""One-command open-source smoke check for AGU."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str]) -> None:
    print(f"==> {name}")
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    run_step("Validate public examples and docs", [sys.executable, "scripts/validate_open_source_baseline.py"])
    run_step("Check CLI entrypoint", [sys.executable, "-m", "app.cli", "--help"])
    run_step("Check package version", [sys.executable, "-m", "app.cli", "--version"])
    run_step("Check plugin diagnostics", [sys.executable, "-m", "app.cli", "plugins", "doctor"])
    run_step("Validate public benchmark", [sys.executable, "scripts/evaluate_public_benchmark.py", "--strict"])
    print("Open-source smoke check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
