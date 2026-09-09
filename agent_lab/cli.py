"""Command-line interface for Agent Lab."""

from __future__ import annotations

import argparse
import sys

from .problems import PROBLEMS
from .evolution import EvolutionEngine


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="agent-lab",
        description="Agent Lab: Evolutionary Program Synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python -m agent_lab
  python -m agent_lab --problem polynomial --seed 42
  python -m agent_lab --problem linear --generations 100 --verbose
  python -m agent_lab --problem absolute --plot
""",
    )
    parser.add_argument(
        "--problem",
        choices=sorted(PROBLEMS.keys()),
        default="polynomial",
        help="target problem to evolve (default: polynomial)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=100,
        help="number of candidates per generation (default: 100)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=250,
        help="maximum number of generations (default: 250)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.25,
        help="probability of mutation per offspring (default: 0.25)",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.70,
        help="probability of crossover vs. cloning (default: 0.70)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="maximum expression tree depth (default: 8)",
    )
    parser.add_argument(
        "--elite-count",
        type=int,
        default=5,
        help="number of elite candidates preserved each generation (default: 5)",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=3,
        help="number of candidates per tournament (default: 3)",
    )
    parser.add_argument(
        "--selection",
        choices=["tournament"],
        default="tournament",
        help="parent selection strategy (default: tournament)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for reproducibility",
    )
    parser.add_argument(
        "--target-fitness",
        type=float,
        default=0.99,
        help="fitness threshold for early termination (default: 0.99)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="generate fitness plot (requires matplotlib)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print detailed progress output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    from .problems import get_problem
    problem = get_problem(args.problem)

    engine = EvolutionEngine(
        problem=problem,
        population_size=args.population,
        max_generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        max_depth=args.max_depth,
        elite_count=args.elite_count,
        tournament_size=args.tournament_size,
        target_fitness=args.target_fitness,
        seed=args.seed,
        verbose=args.verbose,
    )

    metrics = engine.run()

    if args.plot:
        from .visualization import plot_fitness
        output = plot_fitness(metrics)
        if output:
            print(f"\nFitness plot saved to: {output}")
        else:
            print("\nmatplotlib not available -- skipping plot generation.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
