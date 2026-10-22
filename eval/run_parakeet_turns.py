"""ASR candidate: parakeet-mlx decoded per diarization turn instead of per fixed chunk.

The windows come from a diarizer RTTM (senko by default), so this ASR depends on the
diarizer and cannot run in parallel with it. Two ways to build windows:

- ``--cover speech``: only diarized speech. The union of all RTTM segments is merged
  across gaps <= --merge-gap, padded by --pad, and split to <= --max-window seconds.
  Speech the diarizer misses is never decoded.
- ``--cover all``: the whole file, split to <= --max-window seconds. Nothing is skipped;
  the diarizer only picks where to cut.

Splits prefer the middle of a non-speech gap, then a speaker change (any RTTM segment
boundary), and fall back to a hard cut at --max-window. All windows of one meeting are
decoded in one process (parakeet_turns_worker.py), so the model loads once per meeting.

Usage: uv run python run_parakeet_turns.py --name parakeet-v3-turns [--cover speech|all]
       [--max-window 25] [--model M] [--variants sdm,ihm] [--meetings all] [--force]
"""

import argparse
import sys
import tempfile
from pathlib import Path

from common import (
    EVAL_DIR,
    asr_run_dir,
    audio_path,
    diar_run_dir,
    meetings_for,
    read_json,
    read_rttm,
    wav_duration,
    write_json,
)
from measure import measure
from run_parakeet import _package_version, _parakeet_env, to_contract

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


def _union(segments: list[tuple[float, float]], merge_gap: float) -> list[tuple[float, float]]:
    """Merge (start, end) intervals that overlap or are closer than merge_gap."""
    out: list[list[float]] = []
    for start, end in sorted(segments):
        if out and start - out[-1][1] <= merge_gap:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(s, e) for s, e in out]


def _split(
    start: float, end: float, cuts: list[tuple[float, int]], max_window: float, min_window: float
) -> list[tuple[float, float]]:
    """Split [start, end] into windows <= max_window at the best-priority cut points."""
    windows = []
    cur = start
    while end - cur > max_window:
        lo, hi = cur + min_window, cur + max_window
        inside = [(prio, t) for t, prio in cuts if lo <= t <= hi]
        cut = max(inside)[1] if inside else hi  # highest priority, then latest
        windows.append((cur, cut))
        cur = cut
    windows.append((cur, end))
    return windows


def make_windows(
    rttm: list[tuple[str, float, float]],
    duration: float,
    *,
    cover: str,
    max_window: float,
    merge_gap: float,
    pad: float,
) -> list[tuple[float, float]]:
    spans = [(s, e) for _, s, e in rttm]
    speech = _union(spans, 0.0)
    # Cut priority 2: middle of a non-speech gap; 1: any segment boundary (speaker change).
    cuts = [((a_end + b_start) / 2, 2) for (_, a_end), (b_start, _) in zip(speech, speech[1:], strict=False)]
    cuts += [(t, 1) for s, e in spans for t in (s, e)]
    if cover == "all":
        regions = [(0.0, duration)]
    else:
        padded = [(max(0.0, s - pad), min(duration, e + pad)) for s, e in _union(spans, merge_gap)]
        regions = _union(padded, 0.0)
    windows = []
    for start, end in regions:
        windows += _split(start, end, cuts, max_window, max_window / 3)
    return [(round(s, 3), round(e, 3)) for s, e in windows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--diar", default="senko")
    parser.add_argument("--cover", default="speech", choices=["speech", "all"])
    parser.add_argument("--max-window", type=float, default=25.0)
    parser.add_argument("--merge-gap", type=float, default=0.5)
    parser.add_argument("--pad", type=float, default=0.25)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--variants", default="sdm,ihm")
    parser.add_argument("--meetings", default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    env = _parakeet_env()
    settings = {
        "model_id": args.model,
        "package": f"parakeet-mlx=={_package_version(env, 'parakeet-mlx')}",
        "mlx": _package_version(env, "mlx"),
        "windows": {
            "diar": args.diar,
            "cover": args.cover,
            "max_window_s": args.max_window,
            "merge_gap_s": args.merge_gap,
            "pad_s": args.pad,
        },
        "decoding": "greedy, bf16, one window per generate() call",
    }
    for variant in args.variants.split(","):
        out_dir = asr_run_dir(args.name, variant)
        for meeting in meetings_for(args.meetings):
            out = out_dir / f"{meeting}.json"
            if out.exists() and not args.force:
                continue
            wav = audio_path(variant, meeting)
            duration = wav_duration(wav)
            rttm = read_rttm(diar_run_dir(args.diar, variant) / f"{meeting}.rttm")
            windows = make_windows(
                rttm, duration, cover=args.cover, max_window=args.max_window, merge_gap=args.merge_gap, pad=args.pad
            )
            with tempfile.TemporaryDirectory() as tmp:
                windows_path = Path(tmp) / "windows.json"
                write_json(windows_path, windows)
                raw_path = Path(tmp) / "raw.json"
                cmd = [
                    str(env / "bin" / "python"), str(EVAL_DIR / "parakeet_turns_worker.py"),
                    str(wav), str(windows_path), str(raw_path), "--model", args.model,
                ]  # fmt: skip
                meta = measure(cmd, log=out_dir / f"{meeting}.log")
                raw = read_json(raw_path)
            write_json(out_dir / f"{meeting}.raw.json", raw)
            write_json(out_dir / f"{meeting}.windows.json", windows)
            result = to_contract(raw)
            result["meta"] = {
                "model": args.name,
                "settings": settings,
                **meta,
                "audio_s": round(duration, 2),
                "worker_timing": raw.get("timing"),
                "windows_n": len(windows),
                "windows_s": round(sum(e - s for s, e in windows), 1),
            }
            write_json(out, result)
            print(f"{args.name}/{variant}/{meeting}: {meta} windows={len(windows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
