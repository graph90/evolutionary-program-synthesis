"""Tests for the CLI interface."""

import subprocess
import sys


def test_help():
    result = subprocess.run(
        [sys.executable, "-m", "agent_lab", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "Agent Lab" in result.stdout
    assert "--problem" in result.stdout
    assert "--seed" in result.stdout


def test_run_polynomial():
    result = subprocess.run(
        [sys.executable, "-m", "agent_lab", "--problem", "polynomial",
         "--seed", "42", "--generations", "20", "--population", "30"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "EVOLUTION COMPLETE" in result.stdout
    assert "Training fitness" in result.stdout
    assert "Validation fitness" in result.stdout


def test_run_linear():
    result = subprocess.run(
        [sys.executable, "-m", "agent_lab", "--problem", "linear",
         "--seed", "42", "--generations", "20", "--population", "30"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "EVOLUTION COMPLETE" in result.stdout


def test_run_absolute():
    result = subprocess.run(
        [sys.executable, "-m", "agent_lab", "--problem", "absolute",
         "--seed", "42", "--generations", "20", "--population", "30"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "EVOLUTION COMPLETE" in result.stdout


def test_deterministic_seeds():
    r1 = subprocess.run(
        [sys.executable, "-m", "agent_lab", "--problem", "polynomial",
         "--seed", "42", "--generations", "10", "--population", "20"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    r2 = subprocess.run(
        [sys.executable, "-m", "agent_lab", "--problem", "polynomial",
         "--seed", "42", "--generations", "10", "--population", "20"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r1.returncode == 0 and r2.returncode == 0

    def strip_runtime(text: str) -> str:
        return "\n".join(
            line for line in text.splitlines()
            if "Runtime:" not in line
        )

    assert strip_runtime(r1.stdout) == strip_runtime(r2.stdout)
