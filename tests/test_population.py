"""Tests for population generation."""

import random

from agent_lab.population import Candidate, generate_population
from agent_lab.expressions import Constant, Variable, BinaryOp


class TestCandidate:
    def test_creation(self):
        expr = BinaryOp("add", Variable(), Constant(1.0))
        c = Candidate(expression=expr)
        assert c.fitness == 0.0
        assert c.error == float("inf")

    def test_clone(self):
        expr = BinaryOp("add", Variable(), Constant(1.0))
        c = Candidate(expression=expr, fitness=0.5, error=2.0, complexity=3)
        c2 = c.clone()
        assert c2.fitness == 0.5
        assert c2.error == 2.0
        assert c2.complexity == 3
        assert c2 is not c
        assert c2.expression is not c.expression


class TestPopulationGeneration:
    def test_correct_size(self):
        rng = random.Random(42)
        pop = generate_population(rng, size=50, max_depth=4)
        assert len(pop) == 50

    def test_all_candidates_valid(self):
        rng = random.Random(42)
        pop = generate_population(rng, size=20, max_depth=5)
        for c in pop:
            assert c.expression.depth() <= 5
            assert c.complexity == c.expression.size()

    def test_deterministic(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        pop1 = generate_population(rng1, size=10, max_depth=4)
        pop2 = generate_population(rng2, size=10, max_depth=4)
        for c1, c2 in zip(pop1, pop2):
            assert c1.expression.to_string() == c2.expression.to_string()

    def test_different_seeds_different_populations(self):
        rng1 = random.Random(1)
        rng2 = random.Random(2)
        pop1 = generate_population(rng1, size=20, max_depth=4)
        pop2 = generate_population(rng2, size=20, max_depth=4)
        # At least some should differ
        diffs = sum(
            1 for c1, c2 in zip(pop1, pop2)
            if c1.expression.to_string() != c2.expression.to_string()
        )
        assert diffs > 0
