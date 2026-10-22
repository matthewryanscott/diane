"""Transcribe one audio file with mlx-audio's Qwen3-ASR (text only). Runs in envs/qwen3-mlxaudio/.venv.

Usage: .venv/bin/python mlxaudio_worker.py <audio> <out.json> --model M [--chunk-duration 1200] [--max-tokens N]

Writes {"text", "segments": [{text, start, end}] (one per chunk), "timing", "generation_tokens"}.
"""

import argparse
import json
import time

import mlx.core as mx
from mlx_audio.stt.utils import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("out")
    parser.add_argument("--model", required=True)
    parser.add_argument("--chunk-duration", type=float, default=1200.0)
    parser.add_argument("--max-tokens", type=int, default=200_000)  # library default 8192 is a whole-file budget
    parser.add_argument("--language", default="English")
    args = parser.parse_args()

    t0 = time.perf_counter()
    model = load(args.model)
    mx.clear_cache()
    t1 = time.perf_counter()
    out = model.generate(
        args.audio, language=args.language, chunk_duration=args.chunk_duration, max_tokens=args.max_tokens
    )
    t2 = time.perf_counter()
    data = {
        "text": out.text,
        "segments": [{"text": s["text"], "start": s["start"], "end": s["end"]} for s in out.segments],
        "generation_tokens": out.generation_tokens,
        "timing": {"load_s": round(t1 - t0, 2), "transcribe_s": round(t2 - t1, 2)},
        "mlx_peak_memory_mb": round(mx.get_peak_memory() / 2**20, 1),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
