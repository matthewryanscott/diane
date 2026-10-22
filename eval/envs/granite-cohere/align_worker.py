"""Force-align gc_worker.py chunk texts with Qwen3-ForcedAligner-0.6B. Runs in envs/qwen3/.venv (reused read-only).

Usage: ../qwen3/.venv/bin/python align_worker.py <audio> <raw.json> <out.json> [--aligner A] [--dtype float16]

Aligns each chunk's text against that chunk's audio and writes
{"chunks": [{text, start, end, words: [{text, start, end}] (absolute seconds)}], "timing", "mlx_peak_memory_mb"}.
The aligner drops punctuation from its words; the runner maps them back onto the chunk text.
"""

import argparse
import json
import time

import mlx.core as mx
from mlx_qwen3_asr import ForcedAligner
from mlx_qwen3_asr.audio import load_audio

SR = 16000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("raw")
    parser.add_argument("out")
    parser.add_argument("--aligner", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()

    t0 = time.perf_counter()
    aligner = ForcedAligner(args.aligner, dtype=getattr(mx, args.dtype))
    aligner._ensure_loaded()  # noqa: SLF001 - separate load time from alignment
    mx.clear_cache()
    audio = load_audio(args.audio)
    with open(args.raw, encoding="utf-8") as fh:
        chunks = json.load(fh)["chunks"]
    t1 = time.perf_counter()
    out = []
    for chunk in chunks:
        words = []
        if chunk["text"].strip():
            piece = audio[int(chunk["start"] * SR) : int(chunk["end"] * SR)]
            words = [
                {"text": w.text, "start": round(chunk["start"] + w.start_time, 3), "end": round(chunk["start"] + w.end_time, 3)}
                for w in aligner.align(piece, chunk["text"], "English")
            ]  # fmt: skip
        out.append({**chunk, "words": words})
        mx.clear_cache()
    t2 = time.perf_counter()
    data = {
        "chunks": out,
        "aligner": args.aligner,
        "timing": {"load_s": round(t1 - t0, 2), "align_s": round(t2 - t1, 2)},
        "mlx_peak_memory_mb": round(mx.get_peak_memory() / 2**20, 1),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
