"""Population generation and management.

A population is a collection of candidate programs, each containing
an expression tree and metadata used by the evolution engine.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .expressions import Node, generate_random_expression


@dataclass
class Candidate:
    """A single candidate program in the population."""

    expression: Node
    fitness: float = 0.0
    error: float = float("inf")
    complexity: int = 0

    def clone(self) -> Candidate:
        return Candidate(
            expression=self.expression.clone(),
            fitness=self.fitness,
            error=self.error,
            complexity=self.complexity,
        )


def generate_population(
    rng: random.Random,
    size: int,
    max_depth: int,
    constant_range: tuple[float, float] = (-10.0, 10.0),
) -> list[Candidate]:
    """Create an initial population of random candidate programs."""
    population: list[Candidate] = []
    for _ in range(size):
        expr = generate_random_expression(rng, max_depth, constant_range)
        candidate = Candidate(expression=expr, complexity=expr.size())
        population.append(candidate)
    return population
