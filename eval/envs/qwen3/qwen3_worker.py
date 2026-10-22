"""Transcribe one audio file with mlx-qwen3-asr (+ native MLX forced aligner). Runs in envs/qwen3/.venv.

Usage: envs/qwen3/.venv/bin/python qwen3_worker.py <audio> <out.json> --model M [--aligner A] [--dtype float16]
       [--language English] [--no-timestamps]

Writes the library's raw result: {"text", "language", "chunks": [...], "segments": [{text,start,end}...],
"aligned_per_chunk": [n...], "timing": {...}}. Conversion to the eval ASR contract happens in eval/run_qwen3.py.
"""

import argparse
import json
import time

import mlx.core as mx
from mlx_qwen3_asr import ForcedAligner, load_model, transcribe
from mlx_qwen3_asr.audio import load_audio


class _CountingAligner:
    """Pass-through aligner that records how many words each chunk's alignment returned.

    transcribe() flattens all chunks' words into one list; the per-chunk counts let the
    runner map words back onto each chunk's punctuated text.
    """

    def __init__(self, aligner: ForcedAligner) -> None:
        self.aligner = aligner
        self.counts: list[int] = []
        self.seconds = 0.0

    def align(self, audio, text, language):
        t0 = time.perf_counter()
        out = self.aligner.align(audio, text, language)
        self.seconds += time.perf_counter() - t0
        self.counts.append(len(out))
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("out")
    parser.add_argument("--model", required=True)
    parser.add_argument("--aligner", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--language", default="English")
    parser.add_argument("--no-timestamps", action="store_true")
    args = parser.parse_args()

    dtype = getattr(mx, args.dtype)
    t0 = time.perf_counter()
    model, _config = load_model(args.model, dtype=dtype)
    # Loading casts the bf16 checkpoint and leaves the source buffers in MLX's cache (~4 GB
    # for 1.7B); release it so the footprint reflects the steady state, as a real caller would.
    mx.clear_cache()
    aligner = None
    if not args.no_timestamps:
        base = ForcedAligner(args.aligner, dtype=dtype)
        base._ensure_loaded()  # noqa: SLF001 - load now so load time is separated from decode
        mx.clear_cache()
        aligner = _CountingAligner(base)
    t1 = time.perf_counter()
    audio = load_audio(args.audio)
    t2 = time.perf_counter()
    result = transcribe(
        audio,
        model=model,
        language=args.language or None,
        return_timestamps=not args.no_timestamps,
        return_chunks=True,
        forced_aligner=aligner,
    )
    t3 = time.perf_counter()
    data = {
        "text": result.text,
        "language": result.language,
        "finish_reason": result.finish_reason,
        "truncated": result.truncated,
        "chunks": result.chunks,
        "segments": result.segments,
        "aligned_per_chunk": aligner.counts if aligner else None,
        "timing": {
            "load_s": round(t1 - t0, 2),
            "audio_load_s": round(t2 - t1, 2),
            "transcribe_s": round(t3 - t2, 2),
            "align_s": round(aligner.seconds, 2) if aligner else 0.0,
        },
        "mlx_peak_memory_mb": round(mx.get_peak_memory() / 2**20, 1),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
