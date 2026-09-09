"""Selection strategies for choosing parents from the population.

Implements tournament selection and elitism.
"""

from __future__ import annotations

import random

from .population import Candidate


def tournament_select(
    rng: random.Random,
    population: list[Candidate],
    tournament_size: int = 3,
) -> Candidate:
    """Select a single parent via tournament selection.

    Randomly samples `tournament_size` candidates and returns the one
    with the highest fitness.
    """
    if not population:
        raise ValueError("Cannot select from an empty population")

    size = min(tournament_size, len(population))
    competitors = rng.sample(population, size)
    return max(competitors, key=lambda c: c.fitness)


def select_elites(
    population: list[Candidate],
    elite_count: int,
) -> list[Candidate]:
    """Return the top `elite_count` candidates sorted by fitness (descending)."""
    if elite_count <= 0:
        return []
    sorted_pop = sorted(population, key=lambda c: c.fitness, reverse=True)
    return [c.clone() for c in sorted_pop[:elite_count]]
