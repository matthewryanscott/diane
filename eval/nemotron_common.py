"""Shared post-processing for the Nemotron-3-Diarization ports (numpy only).

Both ports emit [T, 8] per-speaker activity probabilities at 10 ms. They go
through the same binarization here, so the two ports differ only in the model
runtime. The defaults match NeMo's e2e_diarize_speech.py without a
post-processing YAML (NVIDIA's card protocol): onset = offset = 0.5, no padding,
no minimum duration.
"""

from pathlib import Path

import numpy as np

FRAME_S = 0.01
THRESHOLD = 0.5


def probs_to_segments(
    probs: np.ndarray, *, threshold: float = THRESHOLD, frame_s: float = FRAME_S, min_dur_s: float = 0.0
) -> list[tuple[str, float, float]]:
    """Binarize [T, S] probabilities into (speaker, start, end) segments sorted by start."""
    segments = []
    for spk in range(probs.shape[1]):
        active = np.concatenate([[False], probs[:, spk] > threshold, [False]])
        edges = np.flatnonzero(np.diff(active.astype(np.int8)))
        for s, e in zip(edges[0::2], edges[1::2], strict=True):
            if (e - s) * frame_s >= min_dur_s:
                segments.append((f"speaker_{spk}", round(s * frame_s, 3), round(e * frame_s, 3)))
    return sorted(segments, key=lambda seg: (seg[1], seg[0]))


def write_rttm(path: Path, file_id: str, segments: list[tuple[str, float, float]]) -> None:
    lines = [
        f"SPEAKER {file_id} 1 {start:.3f} {end - start:.3f} <NA> <NA> {spk} <NA> <NA>"
        for spk, start, end in segments
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
