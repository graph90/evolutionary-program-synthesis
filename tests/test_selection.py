"""Tests for selection strategies."""

import random

from agent_lab.population import Candidate, generate_population
from agent_lab.fitness import evaluate_population
from agent_lab.selection import tournament_select, select_elites
from agent_lab.problems import make_polynomial


class TestTournamentSelect:
    def test_returns_candidate(self):
        rng = random.Random(42)
        pop = generate_population(rng, size=20, max_depth=4)
        problem = make_polynomial()
        evaluate_population(pop, problem)
        selected = tournament_select(rng, pop, tournament_size=3)
        assert isinstance(selected, Candidate)

    def test_prefers_fitter(self):
        rng = random.Random(42)
        fit = Candidate(
            fitness=0.9,
            expression=__import__("agent_lab.expressions", fromlist=["Constant"]).Constant(1.0),
            error=0.1,
        )
        unfit = Candidate(
            fitness=0.1,
            expression=__import__("agent_lab.expressions", fromlist=["Constant"]).Constant(100.0),
            error=99.0,
        )
        # With tournament size = 2 and both always selected, fitter wins
        wins = 0
        for _ in range(100):
            selected = tournament_select(rng, [fit, unfit], tournament_size=2)
            if selected is fit:
                wins += 1
        assert wins > 90


class TestElitism:
    def test_returns_correct_count(self):
        rng = random.Random(42)
        pop = generate_population(rng, size=20, max_depth=4)
        problem = make_polynomial()
        evaluate_population(pop, problem)
        elites = select_elites(pop, elite_count=5)
        assert len(elites) == 5

    def test_elites_are_best(self):
        rng = random.Random(42)
        pop = generate_population(rng, size=20, max_depth=4)
        problem = make_polynomial()
        evaluate_population(pop, problem)
        elites = select_elites(pop, elite_count=3)
        sorted_pop = sorted(pop, key=lambda c: c.fitness, reverse=True)
        for e, p in zip(elites, sorted_pop[:3]):
            assert e.fitness == p.fitness

    def test_zero_elites(self):
        pop = [Candidate(fitness=0.5, expression=__import__("agent_lab.expressions", fromlist=["Variable"]).Variable())]
        assert select_elites(pop, 0) == []

    def test_elites_are_clones(self):
        rng = random.Random(42)
        pop = generate_population(rng, size=10, max_depth=4)
        problem = make_polynomial()
        evaluate_population(pop, problem)
        elites = select_elites(pop, 3)
        # Check they are separate objects
        for e in elites:
            assert e is not pop[0]
