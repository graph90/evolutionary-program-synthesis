"""Expression tree representation for candidate programs.

Candidates are represented as structured expression trees rather than
arbitrary Python code. This ensures safe evaluation with no risk of
arbitrary code execution.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Protected math operations
# ---------------------------------------------------------------------------

def protected_division(a: float, b: float) -> float:
    """Return a / b, or 0.0 when b is zero or the result is non-finite."""
    try:
        result = a / b
        if not math.isfinite(result):
            return 0.0
        return result
    except (ZeroDivisionError, ValueError, OverflowError):
        return 0.0


def protected_log(a: float) -> float:
    """Return log(abs(a)), or 0.0 for non-positive or non-finite input."""
    try:
        result = math.log(abs(a))
        if not math.isfinite(result):
            return 0.0
        return result
    except (ValueError, OverflowError):
        return 0.0


def protected_sqrt(a: float) -> float:
    """Return sqrt(abs(a)), or 0.0 for non-finite input."""
    try:
        result = math.sqrt(abs(a))
        if not math.isfinite(result):
            return 0.0
        return result
    except (ValueError, OverflowError):
        return 0.0


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

@runtime_checkable
class Node(Protocol):
    """Protocol for expression tree nodes."""

    def evaluate(self, x: float) -> float: ...

    def depth(self) -> int: ...

    def size(self) -> int: ...

    def to_string(self) -> str: ...

    def clone(self) -> Node: ...

    def collect_nodes(self) -> list[Node]:
        """Return a flat list of all nodes in this tree."""
        ...


@dataclass(frozen=True)
class Variable:
    """Terminal node representing the input variable x."""

    def evaluate(self, x: float) -> float:
        return x

    def depth(self) -> int:
        return 1

    def size(self) -> int:
        return 1

    def to_string(self) -> str:
        return "x"

    def clone(self) -> Variable:
        return Variable()

    def collect_nodes(self) -> list[Node]:
        return [self]


@dataclass(frozen=True)
class Constant:
    """Terminal node holding a numeric constant."""

    value: float

    def evaluate(self, x: float) -> float:
        return self.value

    def depth(self) -> int:
        return 1

    def size(self) -> int:
        return 1

    def to_string(self) -> str:
        if self.value == int(self.value) and abs(self.value) < 1e12:
            return str(int(self.value))
        return f"{self.value:.4g}"

    def clone(self) -> Constant:
        return Constant(self.value)

    def collect_nodes(self) -> list[Node]:
        return [self]


@dataclass(frozen=True)
class BinaryOp:
    """Binary operator node with two children."""

    operator: str
    left: Node
    right: Node

    def evaluate(self, x: float) -> float:
        left_val = self.left.evaluate(x)
        right_val = self.right.evaluate(x)
        op_func = _BINARY_OPS[self.operator]
        result = op_func(left_val, right_val)
        if not math.isfinite(result):
            return 0.0
        return result

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def to_string(self) -> str:
        left_str = self.left.to_string()
        right_str = self.right.to_string()
        if isinstance(self.left, BinaryOp):
            left_str = f"({left_str})"
        if isinstance(self.right, BinaryOp):
            right_str = f"({right_str})"
        symbol = _BINARY_SYMBOLS[self.operator]
        return f"{left_str} {symbol} {right_str}"

    def clone(self) -> BinaryOp:
        return BinaryOp(self.operator, self.left.clone(), self.right.clone())

    def collect_nodes(self) -> list[Node]:
        result: list[Node] = [self]
        result.extend(self.left.collect_nodes())
        result.extend(self.right.collect_nodes())
        return result


@dataclass(frozen=True)
class UnaryOp:
    """Unary operator node with one child."""

    operator: str
    child: Node

    def evaluate(self, x: float) -> float:
        child_val = self.child.evaluate(x)
        op_func = _UNARY_OPS[self.operator]
        result = op_func(child_val)
        if not math.isfinite(result):
            return 0.0
        return result

    def depth(self) -> int:
        return 1 + self.child.depth()

    def size(self) -> int:
        return 1 + self.child.size()

    def to_string(self) -> str:
        child_str = self.child.to_string()
        if isinstance(self.child, BinaryOp):
            child_str = f"({child_str})"
        if self.operator == "abs":
            return f"|{child_str}|"
        symbol = _UNARY_SYMBOLS[self.operator]
        return f"{symbol}{child_str}"

    def clone(self) -> UnaryOp:
        return UnaryOp(self.operator, self.child.clone())

    def collect_nodes(self) -> list[Node]:
        result: list[Node] = [self]
        result.extend(self.child.collect_nodes())
        return result


# ---------------------------------------------------------------------------
# Operator registries
# ---------------------------------------------------------------------------

_BINARY_OPS: dict[str, callable] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": protected_division,
}

_BINARY_SYMBOLS: dict[str, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}

_UNARY_OPS: dict[str, callable] = {
    "neg": lambda a: -a,
    "abs": lambda a: abs(a),
}

_UNARY_SYMBOLS: dict[str, str] = {
    "neg": "-",
    "abs": "|",
}

BINARY_OPERATORS = list(_BINARY_OPS.keys())
UNARY_OPERATORS = list(_UNARY_OPS.keys())

# Probability of selecting a terminal node during tree generation.
# Used to control tree shape and limit growth.
DEFAULT_TERMINAL_PROBABILITY = 0.3


# ---------------------------------------------------------------------------
# Random expression generation
# ---------------------------------------------------------------------------

def generate_random_expression(
    rng: random.Random,
    max_depth: int,
    constant_range: tuple[float, float] = (-10.0, 10.0),
    terminal_probability: float = DEFAULT_TERMINAL_PROBABILITY,
) -> Node:
    """Generate a random expression tree within the given depth limit.

    Uses the "grow" method: at each level, there is a chance of producing
    a terminal node (variable or constant), which limits tree growth.
    """
    if max_depth <= 1:
        return _generate_terminal(rng, constant_range)

    if rng.random() < terminal_probability:
        return _generate_terminal(rng, constant_range)

    if rng.random() < 0.7:
        op_name = rng.choice(BINARY_OPERATORS)
        left = generate_random_expression(
            rng, max_depth - 1, constant_range, terminal_probability
        )
        right = generate_random_expression(
            rng, max_depth - 1, constant_range, terminal_probability
        )
        return BinaryOp(op_name, left, right)
    else:
        op_name = rng.choice(UNARY_OPERATORS)
        child = generate_random_expression(
            rng, max_depth - 1, constant_range, terminal_probability
        )
        return UnaryOp(op_name, child)


def _generate_terminal(rng: random.Random, constant_range: tuple[float, float]) -> Node:
    """Generate either a variable or a constant terminal."""
    if rng.random() < 0.7:
        return Variable()
    value = rng.uniform(constant_range[0], constant_range[1])
    value = round(value, 2)
    return Constant(value)


# ---------------------------------------------------------------------------
# Random subtree generation (used by mutation and crossover)
# ---------------------------------------------------------------------------

def generate_random_subtree(
    rng: random.Random,
    max_depth: int,
    constant_range: tuple[float, float] = (-10.0, 10.0),
) -> Node:
    """Generate a small random subtree, biased toward terminals."""
    return generate_random_expression(
        rng, max(1, max_depth), constant_range, terminal_probability=0.5
    )
