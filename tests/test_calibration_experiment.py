from __future__ import annotations

import numpy as np
import pandas as pd

from training import calibration_experiment as ce


def _fittable_trend_table(n_subjects: int = 6, n_windows: int = 10) -> tuple[pd.DataFrame, list[str]]:
    """A trend-feature table with signal: eyelid features rise with the label.

    Returns the table and its feature column list (base + trend, minus non-inputs).
    """
    from training import temporal

    rng = np.random.default_rng(0)
    rows = []
    for subject in range(n_subjects):
        label = subject % 3
        video_id = f"{subject:02d}/{label}"
        for window_idx in range(n_windows):
            base = 0.1 + 0.15 * label + 0.01 * window_idx + rng.normal(0, 0.01)
            rows.append(
                {
                    "subject_id": f"{subject:02d}",
                    "video_id": video_id,
                    "window_idx": window_idx,
                    "label": label,
                    "frac_face_missing": 0.0,
                    "bright_mean": 100.0,
                    "bright_std": 1.0,
                    "warmth_mean": 1.0,
                    "warmth_std": 0.1,
                    **{col: base for col in temporal.TREND_COLUMNS},
                    "jaw_open_mean": base,
                    "jaw_open_max": base,
                    "yawn_count": 0,
                    "pitch_mean": base,
                    "pitch_std": 0.1,
                    "pitch_range": 0.2,
                    "frac_head_down": 0.0,
                    "yaw_std": 0.1,
                    "roll_std": 0.1,
                }
            )
    windows_df = pd.DataFrame(rows)
    trend_df = temporal.add_trend_features(windows_df)
    base_columns = temporal.get_sequence_feature_columns(windows_df)
    trend_columns = base_columns + [c for c in trend_df.columns if c not in windows_df.columns]
    return trend_df, trend_columns


def test_softmax_rows_sum_to_one() -> None:
    logits = np.array([[2.0, 1.0, 0.1], [-3.0, 0.0, 4.0]])
    probs = ce._softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs.argmax(axis=1).tolist() == [0, 2]


def test_corn_probabilities_are_rank_monotonic() -> None:
    """CORN's defining property: P(y>=1) >= P(y>=2) for every row, and rows sum to 1."""
    trend_df, trend_columns = _fittable_trend_table()
    corn = ce._fit_corn(trend_df, trend_columns, seed=42)
    probs = ce._predict_corn(corn, trend_df, trend_columns)

    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= -1e-9).all()
    p_ge1 = probs[:, 1] + probs[:, 2]
    p_ge2 = probs[:, 2]
    assert (p_ge1 + 1e-9 >= p_ge2).all()  # cumulative probabilities never increase with rank


def test_fit_temperature_stays_in_bounds() -> None:
    trend_df, trend_columns = _fittable_trend_table(n_subjects=8)
    temperature = ce._fit_temperature(trend_df, trend_columns, seed=42)
    assert 0.05 <= temperature <= 100.0


def test_optimal_temperature_softens_overconfident_logits() -> None:
    """The NLL-minimizing T on an overconfident-but-mostly-right set is >1 and lowers NLL."""
    from scipy.optimize import minimize_scalar

    # Confident and correct on 3/4, confidently wrong on 1 -> softening helps overall NLL.
    logits = np.array([[8.0, 0.0, -8.0], [8.0, 0.0, -8.0], [-8.0, 0.0, 8.0], [-8.0, 0.0, 8.0]])
    labels = np.array([0, 0, 2, 0])
    best_t = float(minimize_scalar(lambda t: ce._nll(logits / t, labels), bounds=(0.05, 100.0), method="bounded").x)

    assert best_t > 1.0  # overconfidence is corrected by softening
    assert ce._nll(logits / best_t, labels) < ce._nll(logits, labels)


def test_temperature_scaling_never_changes_argmax() -> None:
    """Dividing logits by a positive scalar preserves the predicted class."""
    logits = np.array([[2.0, 1.5, 0.1], [-1.0, -0.5, 0.3]])
    for temperature in (0.5, 1.0, 3.0):
        assert ce._softmax(logits / temperature).argmax(axis=1).tolist() == ce._softmax(logits).argmax(axis=1).tolist()


def test_ece_is_zero_for_perfect_calibration() -> None:
    # Confidence exactly matches accuracy in each bin -> ECE 0.
    confidence = np.array([1.0, 1.0, 0.0, 0.0])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    assert ce._ece(confidence, correct) == 0.0
    # A confidently-wrong set has ECE near 1.
    assert ce._ece(np.array([0.95, 0.95]), np.array([0.0, 0.0])) > 0.9


def test_calibration_gap_flags_inversion() -> None:
    by_class = pd.DataFrame(
        [
            {"run": "r", "predicted_class": "low_vigilant", "correct": True, "mean_confidence": 0.44},
            {"run": "r", "predicted_class": "low_vigilant", "correct": False, "mean_confidence": 0.47},
        ]
    )
    gap = ce._calibration_gap(by_class)
    assert np.isclose(gap.loc[0, "gap"], -0.03)  # negative = misses more confident than hits
