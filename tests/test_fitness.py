"""Tests for fitness evaluation."""

import math

from agent_lab.expressions import BinaryOp, Constant, UnaryOp, Variable
from agent_lab.fitness import compute_error, error_to_fitness, evaluate_candidate, evaluate_population
from agent_lab.population import Candidate
from agent_lab.problems import make_linear, make_polynomial


class TestErrorToFitness:
    def test_zero_error(self):
        assert error_to_fitness(0.0) == 1.0

    def test_large_error(self):
        assert error_to_fitness(1000.0) < 0.01

    def test_infinite_error(self):
        assert error_to_fitness(float("inf")) == 0.0

    def test_monotonic(self):
        assert error_to_fitness(0.0) > error_to_fitness(1.0) > error_to_fitness(10.0)


class TestComputeError:
    def test_perfect_solution_polynomial(self):
        # x^2 + 3x + 7
        expr = BinaryOp(
            "add",
            BinaryOp(
                "add",
                BinaryOp("mul", Variable(), Variable()),
                BinaryOp("mul", Constant(3.0), Variable()),
            ),
            Constant(7.0),
        )
        problem = make_polynomial()
        c = Candidate(expression=expr)
        error = compute_error(c, problem)
        assert error < 1e-10

    def test_perfect_solution_linear(self):
        # 4x - 9
        expr = BinaryOp(
            "sub",
            BinaryOp("mul", Constant(4.0), Variable()),
            Constant(9.0),
        )
        problem = make_linear()
        c = Candidate(expression=expr)
        error = compute_error(c, problem)
        assert error < 1e-10

    def test_wrong_solution_worse(self):
        # x (bad for polynomial)
        bad_expr = Variable()
        good_expr = BinaryOp(
            "add",
            BinaryOp(
                "add",
                BinaryOp("mul", Variable(), Variable()),
                BinaryOp("mul", Constant(3.0), Variable()),
            ),
            Constant(7.0),
        )
        problem = make_polynomial()
        bad_error = compute_error(Candidate(expression=bad_expr), problem)
        good_error = compute_error(Candidate(expression=good_expr), problem)
        assert bad_error > good_error


class TestEvaluateCandidate:
    def test_updates_fields(self):
        expr = Variable()
        c = Candidate(expression=expr)
        problem = make_polynomial()
        evaluate_candidate(c, problem)
        assert c.fitness > 0
        assert c.error < float("inf")
        assert c.complexity > 0


class TestEvaluatePopulation:
    def test_all_evaluated(self):
        from agent_lab.population import generate_population
        import random
        rng = random.Random(42)
        pop = generate_population(rng, size=10, max_depth=4)
        problem = make_polynomial()
        evaluate_population(pop, problem)
        for c in pop:
            assert c.fitness >= 0
            assert c.fitness <= 1
            assert c.error < float("inf")
