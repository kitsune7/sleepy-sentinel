from __future__ import annotations

import torch
import pytest

import models


def test_build_cross_entropy_mlp_returns_torch_module_with_three_logits() -> None:
    model = models.build_cross_entropy_mlp(input_dim=4, hidden_dims=(8, 4), dropout=0.25, num_classes=3)

    assert isinstance(model, torch.nn.Module)
    logits = model(torch.zeros((2, 4), dtype=torch.float32))
    assert logits.shape == (2, 3)
    assert any(isinstance(module, torch.nn.Dropout) for module in model.modules())


def test_build_ordinal_mlp_returns_k_minus_one_logits() -> None:
    model = models.build_ordinal_mlp(input_dim=4, hidden_dims=(8,), dropout=0.0, num_classes=3)

    logits = model(torch.zeros((2, 4), dtype=torch.float32))
    assert logits.shape == (2, 2)


def test_prediction_helpers_return_logits_probabilities_and_labels() -> None:
    class FixedModel(torch.nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[1.0, 3.0, 2.0], [4.0, 1.0, 0.0]], dtype=torch.float32)

    batch = torch.zeros((2, 4), dtype=torch.float32)
    model = FixedModel()

    logits = models.predict_logits(model, batch)
    probabilities = models.predict_probabilities(model, batch)
    labels = models.predict_labels(model, batch)

    assert torch.equal(logits, torch.tensor([[1.0, 3.0, 2.0], [4.0, 1.0, 0.0]]))
    assert probabilities.shape == (2, 3)
    assert probabilities.sum(dim=1).tolist() == pytest.approx([1.0, 1.0])
    assert labels.tolist() == [1, 0]
