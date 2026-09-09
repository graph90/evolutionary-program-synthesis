"""Evolution engine - the core evolutionary loop.

Orchestrates the complete process:
  evaluate → rank → elitism → select → crossover → mutate → new generation
"""

from __future__ import annotations

import random
import sys

from .crossover import crossover
from .expressions import generate_random_expression
from .fitness import evaluate_candidate, evaluate_population
from .metrics import RunMetrics
from .mutation import apply_mutation
from .population import Candidate, generate_population
from .problems import Problem
from .selection import select_elites, tournament_select


class EvolutionEngine:
    """Runs the evolutionary optimization loop."""

    def __init__(
        self,
        problem: Problem,
        population_size: int = 100,
        max_generations: int = 250,
        mutation_rate: float = 0.25,
        crossover_rate: float = 0.70,
        max_depth: int = 8,
        elite_count: int = 5,
        tournament_size: int = 3,
        target_fitness: float = 0.99,
        seed: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.problem = problem
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_depth = max_depth
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.target_fitness = target_fitness
        self.verbose = verbose
        self.seed = seed
        self.rng = random.Random(seed)

        self.population: list[Candidate] = []
        self.metrics = RunMetrics()
        self.best_candidate: Candidate | None = None

    def initialize(self) -> None:
        """Create the initial random population."""
        self.population = generate_population(
            self.rng,
            self.population_size,
            self.max_depth,
        )
        evaluate_population(self.population, self.problem)
        self._update_best()

    def _update_best(self) -> None:
        """Track the best candidate in the population."""
        if not self.population:
            return
        current_best = max(self.population, key=lambda c: c.fitness)
        if self.best_candidate is None or current_best.fitness > self.best_candidate.fitness:
            self.best_candidate = current_best.clone()

    def _record_generation(self, gen: int) -> None:
        """Record metrics for the current generation."""
        fitnesses = [c.fitness for c in self.population]
        best = max(fitnesses)
        avg = sum(fitnesses) / len(fitnesses)
        worst = min(fitnesses)
        best_cand = max(self.population, key=lambda c: c.fitness)
        self.metrics.record_generation(
            generation=gen,
            best_fitness=best,
            average_fitness=avg,
            worst_fitness=worst,
            best_error=best_cand.error,
            best_complexity=best_cand.complexity,
            evaluation_count=len(self.population),
        )

    def _build_next_generation(self) -> list[Candidate]:
        """Create the next generation through elitism, selection, crossover, mutation."""
        next_gen: list[Candidate] = []

        # Elitism: preserve top candidates unchanged
        elites = select_elites(self.population, self.elite_count)
        next_gen.extend(elites)

        # Fill the rest via selection, crossover, and mutation
        offspring: list[Candidate] = []
        while len(offspring) < self.population_size - self.elite_count:
            if self.rng.random() < self.crossover_rate:
                parent_a = tournament_select(self.rng, self.population, self.tournament_size)
                parent_b = tournament_select(self.rng, self.population, self.tournament_size)
                child_a, child_b = crossover(self.rng, parent_a, parent_b, self.max_depth)
                offspring.append(child_a)
                if len(offspring) < self.population_size - self.elite_count:
                    offspring.append(child_b)
            else:
                parent = tournament_select(self.rng, self.population, self.tournament_size)
                offspring.append(parent.clone())

        # Apply mutation independently to each offspring
        for i in range(len(offspring)):
            if self.rng.random() < self.mutation_rate:
                offspring[i] = apply_mutation(
                    self.rng, offspring[i], self.max_depth,
                )

        next_gen.extend(offspring[:self.population_size - self.elite_count])
        return next_gen

    def run(self) -> RunMetrics:
        """Execute the complete evolutionary run."""
        self.metrics.start()

        self.initialize()
        self.metrics.record_generation(
            generation=1,
            best_fitness=self.best_candidate.fitness,
            average_fitness=sum(c.fitness for c in self.population) / len(self.population),
            worst_fitness=min(c.fitness for c in self.population),
            best_error=self.best_candidate.error,
            best_complexity=self.best_candidate.complexity,
            evaluation_count=len(self.population),
        )
        self._print_header()
        self._print_progress()

        target_reached = False
        termination_reason = ""
        step = max(1, self.max_generations // 20)

        for gen in range(2, self.max_generations + 1):
            self.population = self._build_next_generation()
            evaluate_population(self.population, self.problem)
            self._update_best()
            self._record_generation(gen)

            reached = bool(
                self.best_candidate and self.best_candidate.fitness >= self.target_fitness
            )
            if self.verbose or gen % step == 0 or gen == self.max_generations or reached:
                self._print_progress()

            if reached:
                target_reached = True
                termination_reason = "TARGET FITNESS REACHED"
                break

        if not target_reached:
            termination_reason = "MAXIMUM GENERATIONS REACHED"

        self.metrics.finish(target_reached, termination_reason)
        self._print_result()
        return self.metrics

    def _print_header(self) -> None:
        """Print the experiment header."""
        print()
        print("=" * 60)
        print(" AGENT LAB")
        print(" EVOLUTIONARY PROGRAM SYNTHESIS")
        print("=" * 60)
        print()
        print(f"Problem:      {self.problem.name}")
        print(f"Target:       {self.problem.description}")
        print(f"Population:   {self.population_size}")
        print(f"Generations:  {self.max_generations}")
        print(f"Mutation:     {self.mutation_rate}")
        print(f"Crossover:    {self.crossover_rate}")
        print(f"Max Depth:    {self.max_depth}")
        print(f"Elites:       {self.elite_count}")
        print(f"Seed:         {self.seed if self.seed is not None else 'random'}")
        print()
        print("-" * 60)
        print(f" {'Gen':>6}  {'Best':>10}  {'Average':>10}  {'Complexity':>10}")
        print("-" * 60)

    def _print_progress(self) -> None:
        """Print the most recent generation's progress."""
        if not self.metrics.generations:
            return
        rec = self.metrics.generations[-1]
        print(f" {rec.generation:>6}  {rec.best_fitness:>10.4f}  {rec.average_fitness:>10.4f}  {rec.best_complexity:>10}")
        if self.verbose and self.best_candidate:
            print(f"   best: {self.best_candidate.expression.to_string()}")

    def _print_result(self) -> None:
        """Print the final result summary."""
        print("-" * 60)
        print()
        print("=" * 60)
        print(" EVOLUTION COMPLETE")
        print("=" * 60)
        print()
        print(f"Problem:           {self.problem.name}")
        print(f"Seed:              {self.seed}")
        print(f"Generations:       {len(self.metrics.generations)}")
        print(f"Runtime:           {self.metrics.elapsed:.2f} seconds")
        print()
        print(f"Termination:       {self.metrics.termination_reason}")
        print()

        if self.best_candidate:
            print("Best expression:")
            print()
            print(f"  {self.best_candidate.expression.to_string()}")
            print()

            # Training fitness
            evaluate_candidate(self.best_candidate, self.problem)
            print(f"Training fitness:  {self.best_candidate.fitness:.6f}")
            print(f"Training error:    {self.best_candidate.error:.6f}")
            print()

            # Validation fitness
            val_error = _compute_validation_error(
                self.best_candidate, self.problem
            )
            val_fitness = 1.0 / (1.0 + max(0.0, val_error))
            print(f"Validation fitness: {val_fitness:.6f}")
            print(f"Validation error:   {val_error:.6f}")
            print()

            print(f"Expression depth:  {self.best_candidate.expression.depth()}")
            print(f"Expression size:   {self.best_candidate.expression.size()}")

        print()
        print("=" * 60)


def _compute_validation_error(candidate: Candidate, problem: Problem) -> float:
    """Compute mean squared error on validation data."""
    if not problem.validation_inputs:
        return 0.0
    total = 0.0
    for x in problem.validation_inputs:
        predicted = problem.evaluate_candidate(candidate.expression, x)
        expected = problem.evaluate_target(x)
        total += (predicted - expected) ** 2
    return total / len(problem.validation_inputs)
