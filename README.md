# Agent Lab
# evolutionary-program-synthesis


Agent Lab demonstrates how a population of random mathematical programs can evolve toward useful solutions using evolutionary computation -- without writing the solution ourselves.


## The Core Idea

The system generates a population of small expression trees, evaluates how well each one approximates a target function, and then uses natural-selection-inspired operators to build the next generation:

```
generate random programs
        ↓
evaluate against target
        ↓
score by fitness
        ↓
select successful programs
        ↓
mutate and recombine
        ↓
repeat
```

Over many generations, fitness improves and the system discovers expressions that approximate the target function -- entirely through evolution.

## Why It Is Interesting

Most programming involves writing a specific solution by hand. Here the system discovers solutions autonomously through a process analogous to biological evolution: random variation, selective pressure, and inheritance of successful traits.

The evolved expressions are not reverse-engineered from the source code. They are genuinely discovered through the evolutionary process. Watching the system converge on a useful solution from random noise makes the underlying search tangible.

## Architecture

```
agent_lab/
├── __init__.py          # Package metadata
├── __main__.py          # Module entry point
├── cli.py               # Argument parsing and CLI interface
├── expressions.py       # Expression tree representation and safe evaluation
├── population.py        # Candidate generation and management
├── fitness.py           # Fitness evaluation
├── selection.py         # Tournament selection and elitism
├── mutation.py          # Point mutation operators
├── crossover.py         # Subtree crossover
├── evolution.py         # Main evolution engine loop
├── problems.py          # Built-in target problems
├── metrics.py           # Run statistics tracking
└── visualization.py     # Optional fitness-over-generations plotting
```

Modules are deliberately small and explicit. Each evolutionary concept -- selection, mutation, crossover, elitism -- lives in its own module with clear, readable code.

## Candidate Representation

Programs are represented as **expression trees**, not as arbitrary executable Python. Each tree consists of:

- **Terminals**: the variable `x` or numeric constants
- **Binary operators**: `+`, `-`, `*`, and protected division
- **Unary operators**: negation and absolute value

Example -- the expression `x * x + 7` is represented as:

```
        ADD
       /   \
     MUL    7
    /   \
   x     x
```

This representation allows safe evaluation without executing arbitrary code.

## Fitness

Candidates are scored using mean squared error against the target function on training points:

```
fitness = 1 / (1 + mean_squared_error)
```

A perfect match produces fitness 1.0. Random programs produce near-zero fitness. Higher fitness always means better performance.

The system maintains both training and validation datasets. Training points guide evolution; validation points are never used for optimization and serve as a generalization check.

## Evolution

### Selection

**Tournament selection** randomly samples a subset of candidates and returns the strongest. The tournament size is configurable.

**Elitism** preserves the top-performing candidates unchanged between generations, ensuring the best solutions are never lost.

### Mutation

Four mutation operators, applied probabilistically:

| Operator | Description |
|----------|-------------|
| Constant | Perturbs a single random constant by a small delta |
| Operator | Swaps one operator node for a compatible alternative |
| Subtree | Replaces a random subtree with a new random expression |
| Variable | Swaps a variable reference for a constant |

Single-node mutations are weighted more heavily than subtree replacements, preserving good structures while allowing refinement.

### Crossover

**Subtree crossover** randomly selects a subtree from each parent and exchanges them, producing two offspring. Crossover respects the maximum depth constraint.

### The Loop

For each generation:

1. Preserve elites
2. Select parents via tournament
3. Apply crossover to create offspring
4. Apply mutation probabilistically
5. Evaluate the new generation
6. Track metrics and check termination

## Safety

Generated programs are never executed as Python source code. There is no `eval()` or `exec()` on generated expressions. The expression tree is evaluated directly through safe, pure-Python evaluation functions.

Candidate programs cannot:
- Execute shell commands
- Access the filesystem
- Access the network
- Import arbitrary modules

Division by zero, overflow, NaN, and infinity are all handled gracefully via protected operations.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

For fitness plots:

```bash
pip install matplotlib
```

## Usage

### Basic Run

```bash
python -m agent_lab
```

### Polynomial Target

```bash
python -m agent_lab --problem polynomial --seed 42
```

### With Full Options

```bash
python -m agent_lab \
    --problem polynomial \
    --population 100 \
    --generations 250 \
    --mutation-rate 0.25 \
    --crossover-rate 0.70 \
    --max-depth 8 \
    --elite-count 5 \
    --seed 42 \
    --verbose
```

### Generate Fitness Plot

```bash
python -m agent_lab --problem polynomial --seed 42 --plot
```

Saves `results/fitness.png`.

### Help

```bash
python -m agent_lab --help
```

## Example Output

Running `python -m agent_lab --problem polynomial --seed 42 --generations 250`:

```
============================================================
 AGENT LAB
 EVOLUTIONARY PROGRAM SYNTHESIS
============================================================

Problem:      polynomial
Target:       f(x) = x^2 + 3x + 7
Population:   100
Generations:  250
Mutation:     0.25
Crossover:    0.7
Max Depth:    8
Elites:       5
Seed:         42

------------------------------------------------------------
    Gen        Best     Average  Complexity
------------------------------------------------------------
      1      0.0120      0.0023          32
     12      0.6819      0.0472          32
     24      0.9955      0.5500          32
------------------------------------------------------------

============================================================
 EVOLUTION COMPLETE
============================================================

Problem:           polynomial
Seed:              42
Generations:       24
Runtime:           0.14 seconds

Termination:       TARGET FITNESS REACHED

Best expression:

  x + |(|(|(x * x)| * ((x - x) + (x / x)))| + (x + (((x / 1.77) + (7.01 - x)) + ((x / 2.42) + x))))|

Training fitness:  0.995541
Training error:    0.004479

Validation fitness: 0.989176
Validation error:   0.010943

Expression depth:  8
Expression size:   32

============================================================
```

This particular seed reaches the target fitness threshold in just 24 generations. Other seeds may take longer or find different expressions that also approximate the polynomial.

## Visualization

When `--plot` is passed and matplotlib is installed, the system generates a fitness-over-generations chart at `results/fitness.png`:

```bash
python -m agent_lab --problem polynomial --seed 42 --plot
```

The plot shows both best fitness and average fitness over the course of evolution.

## Reproducibility

Every run uses a centralized `random.Random` instance seeded at startup. Running the same command with the same `--seed` produces identical results.

```bash
python -m agent_lab --problem polynomial --seed 42
python -m agent_lab --problem polynomial --seed 42  # identical output
```

Omitting `--seed` produces a non-deterministic run.

## Limitations

- The system searches a finite space of expression trees with bounded depth. Expressions beyond the depth limit cannot be discovered.
- Evolved expressions are mathematically equivalent to the target but may look structurally different (e.g., `10 + (-x + x) + x * x` vs. `x * x + 7`).
- Complex problems with many local optima may require larger populations or more generations.
- The current operator set (add, subtract, multiply, protected division, negation, absolute value) limits the kinds of functions that can be discovered.
- No form of bloat control beyond maximum depth. Some evolved expressions are unnecessarily large.

## Future Experiments

Possible extensions to explore:

- Additional operators: `log`, `exp`, `sqrt`, `sin`, `cos`
- Multiple input variables
- Complexity penalties to favor simpler expressions
- Boolean program synthesis
- Island populations with migration
- Parallel evolution across multiple cores
- Alternative selection strategies (rank selection, roulette wheel)
- Novelty search instead of pure fitness optimization
- Coevolution of programs and test cases

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```
