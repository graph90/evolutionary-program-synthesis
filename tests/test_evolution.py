"""Tests for the evolution engine."""

from agent_lab.evolution import EvolutionEngine
from agent_lab.problems import get_problem


class TestEvolutionEngine:
    def test_initialization(self):
        problem = get_problem("linear")
        engine = EvolutionEngine(
            problem=problem,
            population_size=20,
            max_generations=5,
            seed=42,
        )
        engine.initialize()
        assert len(engine.population) == 20
        assert engine.best_candidate is not None

    def test_run_completes(self):
        problem = get_problem("linear")
        engine = EvolutionEngine(
            problem=problem,
            population_size=30,
            max_generations=10,
            seed=42,
        )
        metrics = engine.run()
        assert len(metrics.generations) > 0
        assert metrics.elapsed > 0

    def test_population_size_stable(self):
        problem = get_problem("linear")
        engine = EvolutionEngine(
            problem=problem,
            population_size=50,
            max_generations=5,
            seed=42,
        )
        engine.run()
        assert len(engine.population) == 50

    def test_deterministic(self):
        problem = get_problem("linear")
        m1 = EvolutionEngine(
            problem=problem,
            population_size=30,
            max_generations=10,
            seed=42,
        ).run()
        m2 = EvolutionEngine(
            problem=problem,
            population_size=30,
            max_generations=10,
            seed=42,
        ).run()
        assert len(m1.generations) == len(m2.generations)
        for g1, g2 in zip(m1.generations, m2.generations):
            assert g1.best_fitness == g2.best_fitness

    def test_polynomial_convergence(self):
        """Run longer and check fitness improves significantly."""
        problem = get_problem("polynomial")
        engine = EvolutionEngine(
            problem=problem,
            population_size=100,
            max_generations=100,
            seed=42,
        )
        metrics = engine.run()
        # Should improve from initial
        initial = metrics.generations[0].best_fitness
        final = metrics.generations[-1].best_fitness
        assert final > initial

    def test_linear_convergence(self):
        problem = get_problem("linear")
        engine = EvolutionEngine(
            problem=problem,
            population_size=100,
            max_generations=100,
            seed=42,
        )
        metrics = engine.run()
        initial = metrics.generations[0].best_fitness
        final = metrics.generations[-1].best_fitness
        assert final > initial
