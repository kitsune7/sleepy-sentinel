from __future__ import annotations

import numpy as np
import torch

from training import sequence_models


def test_pad_sequences_pads_and_records_lengths() -> None:
    short = np.ones((3, 4), dtype=np.float32)
    long = np.full((6, 4), 2.0, dtype=np.float32)

    padded, lengths = sequence_models.pad_sequences([short, long])

    assert padded.shape == (2, 6, 4)
    assert lengths.tolist() == [3, 6]
    assert torch.all(padded[0, 3:] == 0.0)


def test_gru_classifier_ignores_padding() -> None:
    """A padded short sequence must score identically to the unpadded one."""
    torch.manual_seed(0)
    model = sequence_models.GruVideoClassifier(input_dim=4, hidden_dim=8, dropout=0.0)
    model.eval()

    sequence = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
    alone, alone_lengths = sequence_models.pad_sequences([sequence])
    with_pad, pad_lengths = sequence_models.pad_sequences(
        [sequence, np.zeros((9, 4), dtype=np.float32)]
    )

    with torch.no_grad():
        logits_alone = model(alone, alone_lengths)
        logits_padded = model(with_pad, pad_lengths)

    assert torch.allclose(logits_alone[0], logits_padded[0], atol=1e-6)
