#!/usr/bin/env python3
"""Repository harness verification for AGU.

This script is intentionally lightweight by default. It checks durable workflow
files, common repository safety issues, Python syntax, and configuration docs.
Pass --run-tests or --test-command to include pytest verification.
"""

from __future__ import annotations

import argparse
import ast
import py_compile
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    ROOT / "AGENTS.md",
    ROOT / "docs/harness/WORKFLOW.md",
    ROOT / "docs/harness/TASK-BOARD.md",
]

WORKFLOW_MARKERS = [
    "### W1 Requirement",
    "### W2 Solution",
    "### W3 Gate Review",
    "### W4 Development",
    "### W5 Code Review",
    "### W6 Testing",
    "## Completion Definition",
]

TASK_BOARD_MARKERS = [
    "## Status Legend",
    "## In Progress",
    "## Completed",
    "## Paused Or Blocked",
    "## Maintenance Rules",
]

GENERATED_PREFIXES = (
    "analysis_outputs/",
    "output_videos/",
    "model_checkpoints/",
    "dataset/",
)


class CheckResult:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def run_command(command: list[str] | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        shell=isinstance(command, str),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def python_with_pytest() -> str | None:
    candidates = [
        ROOT / "venv/bin/python",
        ROOT / ".venv/bin/python",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        proc = subprocess.run(
            [str(candidate), "-c", "import pytest"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode == 0:
            return str(candidate)
    return None


def python_with_module(module: str) -> str:
    candidates = [
        ROOT / "venv/bin/python",
        ROOT / ".venv/bin/python",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        proc = subprocess.run(
            [str(candidate), "-c", f"import {module}"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode == 0:
            return str(candidate)
    return sys.executable


def normalize_test_command(command: str) -> list[str] | str:
    parts = shlex.split(command)
    if parts and parts[0] == "pytest":
        python = python_with_pytest()
        if python:
            return [python, "-m", "pytest", *parts[1:]]
    return command


def check_required_files(result: CheckResult) -> None:
    for path in REQUIRED_FILES:
        if not path.exists():
            result.fail(f"Missing required file: {path.relative_to(ROOT)}")


def check_markers(result: CheckResult) -> None:
    workflow = ROOT / "docs/harness/WORKFLOW.md"
    task_board = ROOT / "docs/harness/TASK-BOARD.md"

    if workflow.exists():
        text = workflow.read_text(encoding="utf-8")
        for marker in WORKFLOW_MARKERS:
            if marker not in text:
                result.fail(f"Missing workflow marker: {marker}")

    if task_board.exists():
        text = task_board.read_text(encoding="utf-8")
        for marker in TASK_BOARD_MARKERS:
            if marker not in text:
                result.fail(f"Missing task board marker: {marker}")


def check_generated_not_staged(result: CheckResult) -> None:
    proc = run_command(["git", "status", "--porcelain"])
    if proc.returncode != 0:
        result.warn("Could not inspect git status.")
        return

    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        status = line[:2]
        path = line[3:]
        if path.startswith(GENERATED_PREFIXES) and status[0] != "?":
            result.fail(f"Generated or large-data path is staged: {path}")


def iter_python_files() -> list[Path]:
    ignored_parts = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "dataset",
        "model_checkpoints",
        "analysis_outputs",
        "output_videos",
    }
    files: list[Path] = []
    for path in ROOT.rglob("*.py"):
        rel_parts = set(path.relative_to(ROOT).parts)
        if rel_parts & ignored_parts:
            continue
        files.append(path)
    return sorted(files)


def check_python_compiles(result: CheckResult) -> None:
    for path in iter_python_files():
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            result.fail(f"Python compile failed for {path.relative_to(ROOT)}: {exc.msg}")


def extract_settings_fields() -> set[str]:
    config_path = ROOT / "app/config.py"
    tree = ast.parse(config_path.read_text(encoding="utf-8"))
    fields: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Settings":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.target.id != "model_config":
                        fields.add(stmt.target.id)
    return fields


def check_env_example(result: CheckResult) -> None:
    env_path = ROOT / ".env.example"
    if not env_path.exists():
        result.fail("Missing .env.example")
        return

    env_text = env_path.read_text(encoding="utf-8")
    missing: list[str] = []
    for field in sorted(extract_settings_fields()):
        env_name = "BASKETBALL_" + field.upper()
        if env_name not in env_text:
            missing.append(env_name)

    if missing:
        joined = ", ".join(missing)
        result.fail(f".env.example is missing Settings variables: {joined}")


def check_open_source_baseline(result: CheckResult) -> None:
    script = ROOT / "scripts/validate_open_source_baseline.py"
    if not script.exists():
        result.fail("Missing scripts/validate_open_source_baseline.py")
        return

    proc = run_command([python_with_module("pydantic"), str(script)])
    if proc.returncode != 0:
        result.fail(proc.stdout)


def run_tests(result: CheckResult, command: str | None, run_all: bool) -> None:
    if not command and not run_all:
        return

    test_command = command or "pytest"
    proc = run_command(normalize_test_command(test_command))
    if proc.returncode != 0:
        result.fail(f"Test command failed: {test_command}\n{proc.stdout}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AGU harness verification.")
    parser.add_argument("--run-tests", action="store_true", help="Run the full pytest suite after structural checks.")
    parser.add_argument("--test-command", default=None, help="Run a focused test command, for example 'pytest tests/test_inference.py'.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = CheckResult()

    check_required_files(result)
    check_markers(result)
    check_generated_not_staged(result)
    check_python_compiles(result)
    check_env_example(result)
    check_open_source_baseline(result)
    run_tests(result, args.test_command, args.run_tests)

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")

    if result.failures:
        print("Harness verification failed:")
        for failure in result.failures:
            print(f"- {failure}")
        return 1

    print("Harness verification passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
