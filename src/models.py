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

import torch


def build_cross_entropy_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
    num_classes: int,
) -> Any:
    """Create the main 3-class MLP baseline."""
    return _build_mlp(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, output_dim=num_classes)


def build_ordinal_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
    num_classes: int,
) -> Any:
    """Create an ordinal model variant."""
    return _build_mlp(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, output_dim=num_classes - 1)


def predict_logits(model: Any, batch: Any) -> Any:
    """Run the model and return raw logits."""
    model.eval()
    with torch.no_grad():
        return model(batch)


def predict_probabilities(model: Any, batch: Any) -> Any:
    """Convert model outputs into class probabilities or ordinal probabilities."""
    return torch.softmax(predict_logits(model, batch), dim=1)


def predict_labels(model: Any, batch: Any) -> Any:
    """Convert model outputs into predicted class labels."""
    return predict_probabilities(model, batch).argmax(dim=1)


def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...], dropout: float, output_dim: int) -> torch.nn.Module:
    layers: list[torch.nn.Module] = []
    current_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(torch.nn.Linear(current_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        current_dim = hidden_dim

    layers.append(torch.nn.Linear(current_dim, output_dim))
    return torch.nn.Sequential(*layers)
