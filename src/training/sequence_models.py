"""Torch-dependent sequence-model pieces for the (rejected) GRU path.

Assignment 8's decision memo promised that the champion path — trend features
plus multinomial logistic regression — stays sklearn-only, with torch optional
for the rejected/diagnostic runs. Assignment 9's failure analysis made that
promise load-bearing: it imports the trend-feature code from
`training.temporal` without needing torch installed. So the two torch users
that used to live in `temporal.py` (sequence padding and the GRU classifier)
moved here, where only `train_temporal` — the A8 experiment runner — imports
them. Behavior is unchanged; only the import boundary moved.
"""

from __future__ import annotations

import numpy as np
import torch


def pad_sequences(sequences: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length `(T, F)` arrays into a zero-padded batch.

    Returns `(padded, lengths)` where `padded` has shape
    `(batch, max_T, F)` and `lengths` holds each sequence's true length so the
    GRU can ignore the padding via packing.
    """
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    max_len = int(lengths.max())
    feature_dim = sequences[0].shape[1]
    padded = torch.zeros((len(sequences), max_len, feature_dim), dtype=torch.float32)
    for idx, seq in enumerate(sequences):
        padded[idx, : len(seq)] = torch.from_numpy(np.ascontiguousarray(seq))
    return padded, lengths


class GruVideoClassifier(torch.nn.Module):
    """Small many-to-one GRU: read a video's window sequence, predict its label."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.25, num_classes: int = 3):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, final_hidden = self.gru(packed)
        return self.head(self.dropout(final_hidden[-1]))
