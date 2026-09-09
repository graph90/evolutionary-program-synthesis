"""Fitness evaluation for candidate programs.

Fitness measures how well a candidate approximates the target function.
Higher fitness always means better performance.
"""

from __future__ import annotations

import math

from .expressions import Node
from .population import Candidate
from .problems import Problem


def compute_error(candidate: Candidate, problem: Problem) -> float:
    """Compute mean squared error of a candidate on the training data."""
    total_error = 0.0
    n = len(problem.training_inputs)
    if n == 0:
        return float("inf")

    for x in problem.training_inputs:
        predicted = problem.evaluate_candidate(candidate.expression, x)
        expected = problem.evaluate_target(x)
        total_error += (predicted - expected) ** 2

    return total_error / n


def error_to_fitness(error: float) -> float:
    """Convert mean squared error to a fitness score in (0, 1].

    fitness = 1 / (1 + error)

    An error of 0 gives fitness 1.0 (perfect).
    Large errors approach fitness 0.0.
    """
    if not math.isfinite(error):
        return 0.0
    return 1.0 / (1.0 + max(0.0, error))


def evaluate_candidate(candidate: Candidate, problem: Problem) -> None:
    """Evaluate a candidate and update its fitness and error fields."""
    error = compute_error(candidate, problem)
    candidate.error = error
    candidate.fitness = error_to_fitness(error)
    candidate.complexity = candidate.expression.size()


def evaluate_population(
    population: list[Candidate], problem: Problem
) -> None:
    """Evaluate every candidate in the population in place."""
    for candidate in population:
        evaluate_candidate(candidate, problem)
