"""Optional visualization using matplotlib.

Generates fitness-over-generations plots when matplotlib is available.
"""

from __future__ import annotations

import os
from pathlib import Path

from .metrics import RunMetrics


def plot_fitness(
    metrics: RunMetrics,
    output_path: str = "results/fitness.png",
) -> str | None:
    """Generate a fitness-over-generations plot.

    Returns the output path on success, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    gens = metrics.generation_numbers
    best = metrics.best_fitness_history
    avg = metrics.average_fitness_history

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, best, label="Best Fitness", color="#2196F3", linewidth=2)
    ax.plot(gens, avg, label="Average Fitness", color="#FF9800", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness", fontsize=12)
    ax.set_title("Evolutionary Program Synthesis", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
