"""Diarize one wav with scribe's senko pipeline. Runs under the main diane venv (has senko).

Usage: <repo>/.venv/bin/python senko_worker.py <wav> <meeting> <out_dir>
Writes <out_dir>/<meeting>.rttm, <meeting>.centroids.npy (+ .speakers.json),
and <meeting>.senko.json (senko's own timing stats).
"""

import json
import sys
from pathlib import Path

import numpy as np
from common import write_rttm

import scribe  # resolved via the sys.path entry common adds


class _Capture:
    """Pass-through diarizer that keeps senko's full result (centroids, timings)."""

    def __init__(self, diarizer: scribe._Diarizer) -> None:  # noqa: SLF001
        self.diarizer = diarizer
        self.result: dict | None = None

    def diarize(self, audio_path: str) -> dict | None:
        self.result = self.diarizer.diarize(audio_path)
        return self.result


def main() -> None:
    wav, meeting, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    capture = _Capture(scribe._create_diarizer())  # noqa: SLF001
    segments = scribe.run_diarization(capture, wav)
    write_rttm(out_dir / f"{meeting}.rttm", meeting, [(s.speaker, s.start, s.end) for s in segments])
    result = capture.result or {}
    centroids = result.get("speaker_centroids")
    if centroids is not None:
        speakers = sorted(centroids) if isinstance(centroids, dict) else None
        array = np.stack([np.asarray(centroids[s]) for s in speakers]) if speakers else np.asarray(centroids)
        np.save(out_dir / f"{meeting}.centroids.npy", array)
        (out_dir / f"{meeting}.speakers.json").write_text(json.dumps(speakers))
    stats = {k: result.get(k) for k in ("timing_stats", "raw_speakers_detected", "merged_speakers_detected")}
    (out_dir / f"{meeting}.senko.json").write_text(json.dumps(stats, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
