"""Tests for mutation operators."""

import math
import random

from agent_lab.expressions import (
    BinaryOp,
    Constant,
    UnaryOp,
    Variable,
    generate_random_expression,
)
from agent_lab.mutation import (
    apply_mutation,
    mutate_constant,
    mutate_operator,
    mutate_subtree,
    mutate_variable,
)
from agent_lab.population import Candidate


class TestMutateConstant:
    def test_constant_mutates(self):
        rng = random.Random(42)
        tree = Constant(5.0)
        mutated = mutate_constant(rng, tree)
        assert isinstance(mutated, Constant)
        # With many tries, should eventually change
        changed = False
        for _ in range(50):
            m = mutate_constant(rng, tree)
            if m.value != 5.0:
                changed = True
                break
        assert changed

    def test_preserves_structure(self):
        rng = random.Random(42)
        tree = BinaryOp("add", Constant(1.0), Constant(2.0))
        mutated = mutate_constant(rng, tree)
        assert isinstance(mutated, BinaryOp)
        assert mutated.operator == "add"


class TestMutateOperator:
    def test_operator_mutates(self):
        rng = random.Random(42)
        tree = BinaryOp("add", Variable(), Variable())
        mutated = mutate_operator(rng, tree)
        assert isinstance(mutated, BinaryOp)

    def test_preserves_terminals(self):
        rng = random.Random(42)
        tree = BinaryOp("add", Variable(), Constant(5.0))
        mutated = mutate_operator(rng, tree)
        assert isinstance(mutated, BinaryOp)
        # Terminals should be Variable and Constant
        assert isinstance(mutated.left, Variable)
        assert isinstance(mutated.right, Constant)


class TestMutateSubtree:
    def test_subtree_mutates(self):
        rng = random.Random(42)
        tree = BinaryOp("add", Variable(), Variable())
        mutated = mutate_subtree(rng, tree, max_depth=5)
        assert mutated.depth() <= 5
        assert mutated.size() >= 1


class TestMutateVariable:
    def test_variable_preserves(self):
        rng = random.Random(42)
        tree = BinaryOp("add", Variable(), Constant(5.0))
        mutated = mutate_variable(rng, tree)
        assert isinstance(mutated, BinaryOp)


class TestApplyMutation:
    def test_returns_candidate(self):
        rng = random.Random(42)
        tree = BinaryOp("add", BinaryOp("mul", Variable(), Variable()), Constant(5.0))
        c = Candidate(expression=tree, complexity=tree.size())
        result = apply_mutation(rng, c, max_depth=6)
        assert isinstance(result, Candidate)
        assert result is not c

    def test_does_not_modify_original(self):
        rng = random.Random(42)
        tree = BinaryOp("add", Variable(), Constant(5.0))
        original_str = tree.to_string()
        c = Candidate(expression=tree, complexity=tree.size())
        apply_mutation(rng, c, max_depth=6)
        assert tree.to_string() == original_str

    def test_all_strategies_produce_valid_output(self):
        rng = random.Random(42)
        tree = BinaryOp("add", BinaryOp("mul", Variable(), Variable()), Constant(3.0))
        c = Candidate(expression=tree, complexity=tree.size())
        for strategy in ["constant", "operator", "subtree", "variable"]:
            result = apply_mutation(rng, c, max_depth=6, strategy=strategy)
            assert isinstance(result, Candidate)
            val = result.expression.evaluate(1.0)
            assert math.isfinite(val)
