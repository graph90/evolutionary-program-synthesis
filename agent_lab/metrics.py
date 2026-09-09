"""Run metrics tracking for evolutionary experiments.

Records per-generation statistics and provides a structure for
post-run analysis and visualization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class GenerationRecord:
    """Statistics for a single generation."""

    generation: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    best_error: float
    best_complexity: int
    evaluation_count: int


@dataclass
class RunMetrics:
    """Complete metrics for an evolutionary run."""

    generations: list[GenerationRecord] = field(default_factory=list)
    total_evaluations: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    target_reached: bool = False
    termination_reason: str = ""

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return self.end_time - self.start_time

    def record_generation(
        self,
        generation: int,
        best_fitness: float,
        average_fitness: float,
        worst_fitness: float,
        best_error: float,
        best_complexity: int,
        evaluation_count: int,
    ) -> None:
        """Record stats for one generation."""
        self.generations.append(
            GenerationRecord(
                generation=generation,
                best_fitness=best_fitness,
                average_fitness=average_fitness,
                worst_fitness=worst_fitness,
                best_error=best_error,
                best_complexity=best_complexity,
                evaluation_count=evaluation_count,
            )
        )
        self.total_evaluations += evaluation_count

    def start(self) -> None:
        """Mark the start of a run."""
        self.start_time = time.perf_counter()

    def finish(self, target_reached: bool, reason: str) -> None:
        """Mark the end of a run."""
        self.end_time = time.perf_counter()
        self.target_reached = target_reached
        self.termination_reason = reason

    @property
    def best_fitness_history(self) -> list[float]:
        """List of best fitness values over generations."""
        return [r.best_fitness for r in self.generations]

    @property
    def average_fitness_history(self) -> list[float]:
        """List of average fitness values over generations."""
        return [r.average_fitness for r in self.generations]

    @property
    def generation_numbers(self) -> list[int]:
        """List of generation numbers."""
        return [r.generation for r in self.generations]
