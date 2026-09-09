"""Subtree crossover for recombining two parent expression trees.

Crossover selects random subtrees from two parents and exchanges them,
respecting the maximum depth constraint.
"""

from __future__ import annotations

import random

from .expressions import BinaryOp, Node, UnaryOp
from .population import Candidate


def _random_node(rng: random.Random, node: Node) -> Node:
    """Select a random node from the expression tree."""
    nodes = node.collect_nodes()
    if not nodes:
        return node
    return rng.choice(nodes)


def _replace_random_subtree(
    rng: random.Random,
    tree: Node,
    replacement: Node,
) -> Node:
    """Replace a random subtree in `tree` with `replacement`."""
    nodes = tree.collect_nodes()
    if not nodes:
        return replacement.clone()

    target = rng.choice(nodes)
    return _swap(tree, target, replacement)


def _swap(tree: Node, target: Node, replacement: Node) -> Node:
    """If tree is the target, return replacement; otherwise recurse."""
    if tree is target:
        return replacement.clone()

    if isinstance(tree, BinaryOp):
        new_left = _swap(tree.left, target, replacement)
        new_right = _swap(tree.right, target, replacement)
        if new_left is tree.left and new_right is tree.right:
            return tree
        return BinaryOp(tree.operator, new_left, new_right)

    if isinstance(tree, UnaryOp):
        new_child = _swap(tree.child, target, replacement)
        if new_child is tree.child:
            return tree
        return UnaryOp(tree.operator, new_child)

    return tree.clone()


def crossover(
    rng: random.Random,
    parent_a: Candidate,
    parent_b: Candidate,
    max_depth: int,
) -> tuple[Candidate, Candidate]:
    """Perform subtree crossover between two parents.

    Returns two offspring. If the offspring exceed max_depth, the
    originals are returned unchanged.
    """
    subtree_a = _random_node(rng, parent_a.expression)
    subtree_b = _random_node(rng, parent_b.expression)

    child_a_expr = _replace_random_subtree(rng, parent_a.expression, subtree_b)
    child_b_expr = _replace_random_subtree(rng, parent_b.expression, subtree_a)

    if child_a_expr.depth() > max_depth or child_b_expr.depth() > max_depth:
        return parent_a.clone(), parent_b.clone()

    return (
        Candidate(expression=child_a_expr, complexity=child_a_expr.size()),
        Candidate(expression=child_b_expr, complexity=child_b_expr.size()),
    )
