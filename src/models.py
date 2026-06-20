"""Define small baseline models for alertness classification.

This file owns the model definitions for Stage 5:
- create a simple MLP that predicts the three alertness classes
- optionally create an ordinal/CORN-style variant later
- keep model architecture choices small and easy to compare
- expose prediction helpers used by training and evaluation

For Assignment 5, the most important model is a straightforward baseline plus
one regularized version for the generalization intervention.
"""

from __future__ import annotations

from typing import Any


def build_cross_entropy_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
    num_classes: int,
) -> Any:
    """Create the main 3-class MLP baseline."""
    raise NotImplementedError


def build_ordinal_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
    num_classes: int,
) -> Any:
    """Create an ordinal model variant if you choose to implement CORN later."""
    raise NotImplementedError


def predict_logits(model: Any, batch: Any) -> Any:
    """Run the model and return raw logits."""
    raise NotImplementedError


def predict_probabilities(model: Any, batch: Any) -> Any:
    """Convert model outputs into class probabilities or ordinal probabilities."""
    raise NotImplementedError


def predict_labels(model: Any, batch: Any) -> Any:
    """Convert model outputs into predicted class labels."""
    raise NotImplementedError
