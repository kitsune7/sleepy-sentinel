from __future__ import annotations

from data_prep.extract_features import monotonic_timestamp_ms


def test_monotonic_timestamp_ms_uses_frame_time_when_video_timestamp_is_missing() -> None:
    assert monotonic_timestamp_ms(raw_t_ms=0.0, frame_idx=3, fps=30.0, previous_t_ms=None) == 100


def test_monotonic_timestamp_ms_repairs_repeated_or_backwards_video_timestamps() -> None:
    assert monotonic_timestamp_ms(raw_t_ms=100.0, frame_idx=3, fps=30.0, previous_t_ms=100) == 101
    assert monotonic_timestamp_ms(raw_t_ms=99.0, frame_idx=3, fps=30.0, previous_t_ms=101) == 102


def test_monotonic_timestamp_ms_repairs_rounded_duplicate_timestamps() -> None:
    assert monotonic_timestamp_ms(raw_t_ms=100.2, frame_idx=3, fps=30.0, previous_t_ms=100) == 101


def test_monotonic_timestamp_ms_uses_frame_time_when_video_timestamp_is_nan() -> None:
    assert monotonic_timestamp_ms(raw_t_ms=float("nan"), frame_idx=3, fps=30.0, previous_t_ms=None) == 100
