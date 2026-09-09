"""Mutation operators for evolving expression trees.

Each mutation operator takes a candidate and returns a new mutated
candidate without modifying the original.

Mutations are designed to be *point* mutations: each operator changes
exactly one node in the tree, so a good candidate is nudged rather than
destroyed.
"""

from __future__ import annotations

import random

from .expressions import (
    BINARY_OPERATORS,
    UNARY_OPERATORS,
    BinaryOp,
    Constant,
    Node,
    UnaryOp,
    Variable,
    generate_random_expression,
)
from .population import Candidate


def _depth_from_root(tree: Node, target: Node) -> int:
    """Return the 1-based depth of `target` within `tree` (or 0 if absent)."""
    if tree is target:
        return 1
    if isinstance(tree, BinaryOp):
        for child in (tree.left, tree.right):
            d = _depth_from_root(child, target)
            if d:
                return 1 + d
    elif isinstance(tree, UnaryOp):
        d = _depth_from_root(tree.child, target)
        if d:
            return 1 + d
    return 0


def _replace_node(tree: Node, target: Node, replacement: Node) -> Node:
    """Replace `target` with `replacement`, constructing a new tree."""
    if tree is target:
        return replacement.clone()

    if isinstance(tree, BinaryOp):
        new_left = _replace_node(tree.left, target, replacement)
        new_right = _replace_node(tree.right, target, replacement)
        if new_left is tree.left and new_right is tree.right:
            return tree
        return BinaryOp(tree.operator, new_left, new_right)

    if isinstance(tree, UnaryOp):
        new_child = _replace_node(tree.child, target, replacement)
        if new_child is tree.child:
            return tree
        return UnaryOp(tree.operator, new_child)

    return tree.clone()


def mutate_constant(
    rng: random.Random,
    node: Node,
    constant_range: tuple[float, float] = (-10.0, 10.0),
) -> Node:
    """Return a copy with a single random constant perturbed."""
    constants = [n for n in node.collect_nodes() if isinstance(n, Constant)]
    if not constants:
        return node.clone()

    target = rng.choice(constants)
    delta = rng.uniform(-2.0, 2.0)
    new_val = target.value + delta
    new_val = max(constant_range[0], min(constant_range[1], new_val))
    new_val = round(new_val, 2)
    return _replace_node(node, target, Constant(new_val))


def mutate_operator(rng: random.Random, node: Node) -> Node:
    """Return a copy with a single random operator replaced."""
    bin_ops = [n for n in node.collect_nodes() if isinstance(n, BinaryOp)]
    unary_ops = [n for n in node.collect_nodes() if isinstance(n, UnaryOp)]
    operators = bin_ops + unary_ops
    if not operators:
        return node.clone()

    target = rng.choice(operators)
    if isinstance(target, BinaryOp):
        replacement = BinaryOp(
            rng.choice(BINARY_OPERATORS),
            target.left.clone(),
            target.right.clone(),
        )
    else:
        replacement = UnaryOp(
            rng.choice(UNARY_OPERATORS),
            target.child.clone(),
        )
    return _replace_node(node, target, replacement)


def mutate_subtree(
    rng: random.Random,
    node: Node,
    max_depth: int,
    constant_range: tuple[float, float] = (-10.0, 10.0),
) -> Node:
    """Replace a random subtree with a new random subtree.

    The replacement is generated with a depth limit that guarantees the
    resulting tree does not exceed `max_depth`.
    """
    nodes = node.collect_nodes()
    if not nodes:
        return generate_random_expression(rng, max_depth, constant_range)

    target = rng.choice(nodes)
    target_depth = _depth_from_root(node, target)
    # Remaining height allowed for the replacement so the total stays in bounds.
    allowed = max(1, max_depth - target_depth + 1)
    replacement = generate_random_expression(rng, allowed, constant_range)
    return _replace_node(node, target, replacement)


def mutate_variable(rng: random.Random, node: Node) -> Node:
    """Replace a random variable with a constant from the allowed range."""
    variables = [n for n in node.collect_nodes() if isinstance(n, Variable)]
    if not variables:
        return node.clone()
    target = rng.choice(variables)
    value = round(rng.uniform(-5.0, 5.0), 2)
    return _replace_node(node, target, Constant(value))


MUTATION_OPERATORS = [
    "constant",
    "operator",
    "subtree",
    "variable",
]

# Mutations that change a single node are more likely to preserve and
# refine good structures.
MUTATION_WEIGHTS = {
    "constant": 4,
    "operator": 2,
    "subtree": 2,
    "variable": 1,
}


def apply_mutation(
    rng: random.Random,
    candidate: Candidate,
    max_depth: int,
    constant_range: tuple[float, float] = (-10.0, 10.0),
    strategy: str | None = None,
) -> Candidate:
    """Apply a random mutation to a candidate and return the mutated copy."""
    if strategy is None:
        strategy = rng.choices(
            list(MUTATION_OPERATORS),
            weights=[MUTATION_WEIGHTS[op] for op in MUTATION_OPERATORS],
            k=1,
        )[0]

    expr = candidate.expression
    if strategy == "constant":
        new_expr = mutate_constant(rng, expr, constant_range)
    elif strategy == "operator":
        new_expr = mutate_operator(rng, expr)
    elif strategy == "subtree":
        new_expr = mutate_subtree(rng, expr, max_depth, constant_range)
    elif strategy == "variable":
        new_expr = mutate_variable(rng, expr)
    else:
        new_expr = expr.clone()

    return Candidate(expression=new_expr, complexity=new_expr.size())