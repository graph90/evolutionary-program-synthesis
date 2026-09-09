"""Tests for subtree crossover."""

import math
import random

from agent_lab.expressions import BinaryOp, Constant, Variable
from agent_lab.crossover import crossover
from agent_lab.population import Candidate


class TestCrossover:
    def test_returns_two_offspring(self):
        rng = random.Random(42)
        a = Candidate(
            expression=BinaryOp("add", Variable(), Constant(1.0)),
            complexity=3,
        )
        b = Candidate(
            expression=BinaryOp("mul", Variable(), Constant(2.0)),
            complexity=3,
        )
        child_a, child_b = crossover(rng, a, b, max_depth=6)
        assert isinstance(child_a, Candidate)
        assert isinstance(child_b, Candidate)

    def test_offspring_are_valid(self):
        rng = random.Random(42)
        a = Candidate(
            expression=BinaryOp("add", Variable(), Constant(1.0)),
            complexity=3,
        )
        b = Candidate(
            expression=BinaryOp("mul", Variable(), Constant(2.0)),
            complexity=3,
        )
        for _ in range(20):
            child_a, child_b = crossover(rng, a, b, max_depth=6)
            assert child_a.expression.depth() <= 6
            assert child_b.expression.depth() <= 6
            val_a = child_a.expression.evaluate(1.0)
            val_b = child_b.expression.evaluate(1.0)
            assert math.isfinite(val_a)
            assert math.isfinite(val_b)

    def test_does_not_modify_parents(self):
        rng = random.Random(42)
        expr_a = BinaryOp("add", Variable(), Constant(1.0))
        expr_b = BinaryOp("mul", Variable(), Constant(2.0))
        a = Candidate(expression=expr_a, complexity=3)
        b = Candidate(expression=expr_b, complexity=3)
        str_a = expr_a.to_string()
        str_b = expr_b.to_string()
        crossover(rng, a, b, max_depth=6)
        assert expr_a.to_string() == str_a
        assert expr_b.to_string() == str_b

    def test_respects_max_depth(self):
        rng = random.Random(42)
        for _ in range(30):
            a = Candidate(
                expression=BinaryOp("add", Variable(), Constant(1.0)),
                complexity=3,
            )
            b = Candidate(
                expression=BinaryOp("mul", Variable(), Constant(2.0)),
                complexity=3,
            )
            child_a, child_b = crossover(rng, a, b, max_depth=4)
            assert child_a.expression.depth() <= 4
            assert child_b.expression.depth() <= 4

    def test_deep_parents_fallback(self):
        """Crossover with very deep parents should fallback to originals."""
        rng = random.Random(42)
        a = Candidate(
            expression=BinaryOp("add", Variable(), Constant(1.0)),
            complexity=3,
        )
        b = Candidate(
            expression=BinaryOp("mul", Variable(), Constant(2.0)),
            complexity=3,
        )
        # max_depth=1 means any crossover would be too deep
        child_a, child_b = crossover(rng, a, b, max_depth=1)
        # Should return originals as fallback
        assert child_a.expression.to_string() == a.expression.to_string()
        assert child_b.expression.to_string() == b.expression.to_string()
