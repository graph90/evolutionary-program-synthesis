"""Built-in target problems for evolution.

Each problem provides a target function, training inputs, validation inputs,
and a common interface for the evolution engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .expressions import Node


@dataclass(frozen=True)
class Problem:
    """A regression problem that evolution will attempt to solve."""

    name: str
    description: str
    target_function: callable
    training_inputs: list[float] = field(default_factory=list)
    validation_inputs: list[float] = field(default_factory=list)

    def evaluate_target(self, x: float) -> float:
        """Evaluate the target function at a single point."""
        try:
            result = self.target_function(x)
            if not math.isfinite(result):
                return 0.0
            return result
        except (ValueError, OverflowError):
            return 0.0

    def evaluate_candidate(self, candidate: Node, x: float) -> float:
        """Evaluate a candidate expression at a single point."""
        try:
            result = candidate.evaluate(x)
            if not math.isfinite(result):
                return 0.0
            return result
        except (ValueError, OverflowError):
            return 0.0


def _generate_inputs(low: float, high: float, count: int) -> list[float]:
    """Generate evenly-spaced inputs over a range."""
    step = (high - low) / (count - 1)
    return [round(low + i * step, 4) for i in range(count)]


def make_polynomial() -> Problem:
    """f(x) = x^2 + 3x + 7"""

    def target(x: float) -> float:
        return x * x + 3 * x + 7

    return Problem(
        name="polynomial",
        description="f(x) = x^2 + 3x + 7",
        target_function=target,
        training_inputs=_generate_inputs(-5.0, 5.0, 20),
        validation_inputs=_generate_inputs(-8.0, 8.0, 30),
    )


def make_linear() -> Problem:
    """f(x) = 4x - 9"""

    def target(x: float) -> float:
        return 4 * x - 9

    return Problem(
        name="linear",
        description="f(x) = 4x - 9",
        target_function=target,
        training_inputs=_generate_inputs(-10.0, 10.0, 20),
        validation_inputs=_generate_inputs(-15.0, 15.0, 30),
    )


def make_absolute() -> Problem:
    """f(x) = |x|"""

    def target(x: float) -> float:
        return abs(x)

    return Problem(
        name="absolute",
        description="f(x) = |x|",
        target_function=target,
        training_inputs=_generate_inputs(-10.0, 10.0, 20),
        validation_inputs=_generate_inputs(-15.0, 15.0, 30),
    )


PROBLEMS: dict[str, callable] = {
    "polynomial": make_polynomial,
    "linear": make_linear,
    "absolute": make_absolute,
}


def get_problem(name: str) -> Problem:
    """Return a problem by name."""
    if name not in PROBLEMS:
        available = ", ".join(sorted(PROBLEMS.keys()))
        raise ValueError(f"Unknown problem '{name}'. Available: {available}")
    return PROBLEMS[name]()
