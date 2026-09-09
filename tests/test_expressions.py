"""Tests for expression tree representation and evaluation."""

import math
import random

from agent_lab.expressions import (
    Constant,
    Variable,
    BinaryOp,
    UnaryOp,
    protected_division,
    protected_log,
    protected_sqrt,
    generate_random_expression,
    generate_random_subtree,
)


class TestTerminalNodes:
    def test_variable_evaluation(self):
        x = Variable()
        assert x.evaluate(0) == 0
        assert x.evaluate(5) == 5
        assert x.evaluate(-3) == -3

    def test_constant_evaluation(self):
        c = Constant(7.0)
        assert c.evaluate(0) == 7.0
        assert c.evaluate(999) == 7.0

    def test_constant_to_string_int(self):
        c = Constant(3.0)
        assert c.to_string() == "3"

    def test_constant_to_string_float(self):
        c = Constant(3.14)
        assert c.to_string() == "3.14"

    def test_variable_depth(self):
        assert Variable().depth() == 1

    def test_constant_depth(self):
        assert Constant(5.0).depth() == 1

    def test_variable_size(self):
        assert Variable().size() == 1

    def test_constant_size(self):
        assert Constant(5.0).size() == 1

    def test_variable_clone(self):
        v = Variable()
        c = v.clone()
        assert type(c) is Variable
        assert c is not v

    def test_constant_clone(self):
        c1 = Constant(5.0)
        c2 = c1.clone()
        assert type(c2) is Constant
        assert c2.value == 5.0
        assert c2 is not c1


class TestBinaryOps:
    def test_add(self):
        tree = BinaryOp("add", Variable(), Constant(3.0))
        assert tree.evaluate(2) == 5.0

    def test_sub(self):
        tree = BinaryOp("sub", Variable(), Constant(1.0))
        assert tree.evaluate(5) == 4.0

    def test_mul(self):
        tree = BinaryOp("mul", Variable(), Constant(3.0))
        assert tree.evaluate(4) == 12.0

    def test_div(self):
        tree = BinaryOp("div", Variable(), Constant(2.0))
        assert tree.evaluate(10) == 5.0

    def test_div_by_zero(self):
        tree = BinaryOp("div", Constant(1.0), Constant(0.0))
        assert tree.evaluate(0) == 0.0

    def test_depth(self):
        tree = BinaryOp("add", Variable(), Constant(1.0))
        assert tree.depth() == 2

    def test_nested_depth(self):
        tree = BinaryOp(
            "add",
            BinaryOp("mul", Variable(), Variable()),
            Constant(1.0),
        )
        assert tree.depth() == 3

    def test_size(self):
        tree = BinaryOp("add", Variable(), Constant(1.0))
        assert tree.size() == 3

    def test_to_string(self):
        tree = BinaryOp("add", Variable(), Constant(1.0))
        assert tree.to_string() == "x + 1"

    def test_nested_to_string(self):
        tree = BinaryOp(
            "add",
            BinaryOp("mul", Variable(), Variable()),
            Constant(1.0),
        )
        assert tree.to_string() == "(x * x) + 1"

    def test_clone(self):
        tree = BinaryOp("add", Variable(), Constant(1.0))
        c = tree.clone()
        assert c is not tree
        assert c.to_string() == tree.to_string()


class TestUnaryOps:
    def test_neg(self):
        tree = UnaryOp("neg", Constant(5.0))
        assert tree.evaluate(0) == -5.0

    def test_abs_positive(self):
        tree = UnaryOp("abs", Constant(5.0))
        assert tree.evaluate(0) == 5.0

    def test_abs_negative(self):
        tree = UnaryOp("abs", Constant(-5.0))
        assert tree.evaluate(0) == 5.0

    def test_depth(self):
        tree = UnaryOp("neg", Variable())
        assert tree.depth() == 2

    def test_size(self):
        tree = UnaryOp("abs", Variable())
        assert tree.size() == 2

    def test_to_string_neg(self):
        tree = UnaryOp("neg", Variable())
        assert tree.to_string() == "-x"

    def test_to_string_abs(self):
        tree = UnaryOp("abs", Variable())
        assert tree.to_string() == "|x|"

    def test_clone(self):
        tree = UnaryOp("neg", Constant(3.0))
        c = tree.clone()
        assert c is not tree
        assert c.to_string() == tree.to_string()


class TestProtectedOperations:
    def test_div_normal(self):
        assert protected_division(10.0, 2.0) == 5.0

    def test_div_by_zero(self):
        assert protected_division(10.0, 0.0) == 0.0

    def test_div_neg_by_zero(self):
        assert protected_division(-10.0, 0.0) == 0.0

    def test_log_positive(self):
        result = protected_log(10.0)
        assert abs(result - math.log(10)) < 1e-10

    def test_log_zero(self):
        result = protected_log(0.0)
        assert result == 0.0 or math.isfinite(result)

    def test_log_negative(self):
        result = protected_log(-5.0)
        assert math.isfinite(result)

    def test_sqrt_positive(self):
        result = protected_sqrt(9.0)
        assert abs(result - 3.0) < 1e-10

    def test_sqrt_negative(self):
        result = protected_sqrt(-4.0)
        assert math.isfinite(result)

    def test_complex_nested(self):
        tree = BinaryOp(
            "div",
            BinaryOp("add", Constant(1.0), Constant(2.0)),
            BinaryOp("sub", Constant(1.0), Constant(1.0)),
        )
        assert tree.evaluate(0) == 0.0


class TestCollectNodes:
    def test_single_node(self):
        v = Variable()
        assert v.collect_nodes() == [v]

    def test_binary_tree(self):
        tree = BinaryOp("add", Variable(), Constant(5.0))
        nodes = tree.collect_nodes()
        assert len(nodes) == 3

    def test_unary_tree(self):
        tree = UnaryOp("neg", Variable())
        nodes = tree.collect_nodes()
        assert len(nodes) == 2


class TestRandomGeneration:
    def test_generation_deterministic(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        tree1 = generate_random_expression(rng1, max_depth=5)
        tree2 = generate_random_expression(rng2, max_depth=5)
        assert tree1.to_string() == tree2.to_string()

    def test_generation_respects_depth(self):
        rng = random.Random(42)
        for _ in range(50):
            tree = generate_random_expression(rng, max_depth=3)
            assert tree.depth() <= 3

    def test_generation_depth_1(self):
        rng = random.Random(42)
        tree = generate_random_expression(rng, max_depth=1)
        assert tree.depth() == 1

    def test_subtree_generation(self):
        rng = random.Random(42)
        tree = generate_random_subtree(rng, max_depth=3)
        assert tree.depth() <= 3
        assert tree.size() >= 1

    def test_various_seeds(self):
        for seed in range(20):
            rng = random.Random(seed)
            tree = generate_random_expression(rng, max_depth=6)
            assert tree.depth() <= 6
            val = tree.evaluate(1.0)
            assert math.isfinite(val)
