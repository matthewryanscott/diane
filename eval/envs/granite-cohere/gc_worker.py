"""Transcribe one audio file with Granite-speech-4.1 or Cohere Transcribe via mlx-audio. Runs in envs/granite-cohere/.venv.

Usage: .venv/bin/python gc_worker.py {granite,cohere} <audio> <out.json> --model M [options]

granite: mlx-audio's Granite generate() takes the whole input in one pass, so this
worker cuts the audio itself at the quietest 100 ms window between --min-chunk and
--max-chunk seconds (default 20-30 s) and decodes each chunk greedily with --prompt.
The encoder's relative-position term is computed with an einsum (see
_patch_granite_pos_attn); --orig-pos-attn keeps the library's memory-hungry version.

cohere: mlx-audio's Cohere generate() chunks internally (<=35 s, cut at the quietest
100 ms window in the last 5 s). Decoding is greedy.

Writes {"chunks": [{text, start, end, tokens}], "timing", "mlx_peak_memory_mb", "settings"}.
"""

import argparse
import json
import math
import time

import mlx.core as mx
import numpy as np
from mlx_audio.stt.utils import load, load_audio

SR = 16000
GRANITE_RAW_PROMPT = "can you transcribe the speech into a written format?"


def energy_chunks(audio: np.ndarray, min_s: float, max_s: float, window_s: float = 0.1) -> list[tuple[int, int]]:
    """Cut at the lowest-energy window in [start+min_s, start+max_s]; the last chunk may be shorter."""
    win = int(window_s * SR)
    total = len(audio)
    chunks = []
    start = 0
    while total - start > max_s * SR:
        lo, hi = start + int(min_s * SR), start + int(max_s * SR)
        seg = audio[lo:hi]
        n = len(seg) // win
        energy = (seg[: n * win].reshape(n, win) ** 2).mean(axis=1)
        cut = lo + int(np.argmin(energy)) * win + win // 2
        chunks.append((start, cut))
        start = cut
    chunks.append((start, total))
    return chunks


def _patch_granite_pos_attn() -> None:
    """Compute Granite's relative-position attention term with einsum instead of a broadcast product.

    mlx-audio 0.5.6 materializes q[..., None, :] * rel_pos_emb, a (blocks, heads, 200, 200, 128)
    tensor per layer (~1.3 GB for a 30 s chunk), which puts the process footprint near 10 GB.
    The einsum is the same sum without that intermediate.
    """
    from mlx_audio.stt.models.granite_speech import granite_speech as gs

    def call(self, x: mx.array, attention_dists: mx.array) -> mx.array:
        x = self.pre_norm(x)
        B, N, _ = x.shape
        C = self.context_size
        num_blocks = math.ceil(N / C)
        remainder = N % C
        if remainder > 0:
            x = mx.pad(x, [(0, 0), (0, C - remainder), (0, 0)])
        k, v = mx.split(self.to_kv(x), 2, axis=-1)
        q, k, v = (t.reshape(B, num_blocks, C, self.num_heads, -1).transpose(0, 1, 3, 2, 4) for t in (self.to_q(x), k, v))
        rel = self.rel_pos_emb(attention_dists)
        pos_attn = mx.einsum("bnhid,ijd->bnhij", q, rel.astype(q.dtype)) * self.scale
        if remainder > 0:
            mask = ~((mx.arange(C)[:, None] < remainder) & (mx.arange(C)[None, :] < remainder))
            last = mx.where(mask[None, None, None], mx.array(mx.finfo(pos_attn.dtype).min), pos_attn[:, -1:])
            pos_attn = mx.concatenate([pos_attn[:, :-1], last], axis=1)
        attn = mx.softmax((q @ k.transpose(0, 1, 2, 4, 3)) * self.scale + pos_attn, axis=-1)
        out = (attn @ v).transpose(0, 1, 3, 2, 4).reshape(B, -1, self.num_heads * self.dim_head)
        return self.to_out(out[:, :N, :])

    gs.ConformerAttention.__call__ = call


def run_granite(model, audio: np.ndarray, args) -> list[dict]:
    out = []
    for start, end in energy_chunks(audio, args.min_chunk, args.max_chunk):
        res = model.generate(mx.array(audio[start:end]), prompt=args.prompt, max_tokens=args.max_tokens)
        out.append({"text": res.text.strip(), "start": start / SR, "end": end / SR, "tokens": res.generation_tokens})
        mx.clear_cache()
    return out


def run_cohere(model, audio: np.ndarray, args) -> list[dict]:
    if args.max_clip is not None:  # library default 35 s, split searched in the last 5 s
        model.config.max_audio_clip_s = args.max_clip
    # The optional MLX Silero conversion has no supplied license grant; keep VAD disabled.
    res = model.generate(
        audio, language="en", punctuation=not args.no_punctuation, batch_size=args.batch_size,
        max_tokens=args.max_tokens, vad=False, sample_rate=SR,
    )  # fmt: skip
    return [{"text": s["text"].strip(), "start": s["start"], "end": s["end"], "tokens": None} for s in res.segments]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["granite", "cohere"])
    parser.add_argument("audio")
    parser.add_argument("out")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=GRANITE_RAW_PROMPT)
    parser.add_argument("--min-chunk", type=float, default=20.0)
    parser.add_argument("--max-chunk", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=None, help="per chunk (granite 400, cohere 256)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-clip", type=float, default=None, help="cohere energy-chunk length (default 35 s)")
    parser.add_argument("--no-punctuation", action="store_true")
    parser.add_argument("--orig-pos-attn", action="store_true", help="granite: keep mlx-audio's broadcast pos-attn")
    args = parser.parse_args()
    if args.max_tokens is None:
        args.max_tokens = 400 if args.kind == "granite" else 256

    if args.kind == "granite" and not args.orig_pos_attn:
        _patch_granite_pos_attn()
    t0 = time.perf_counter()
    model = load(args.model)
    mx.clear_cache()
    audio = np.array(load_audio(args.audio), dtype=np.float32)
    t1 = time.perf_counter()
    chunks = (run_granite if args.kind == "granite" else run_cohere)(model, audio, args)
    t2 = time.perf_counter()
    settings = {k: v for k, v in vars(args).items() if k not in ("audio", "out")}
    if args.kind == "cohere":
        for k in ("prompt", "min_chunk", "max_chunk", "orig_pos_attn"):
            settings.pop(k)
    else:
        for k in ("batch_size", "max_clip", "no_punctuation"):
            settings.pop(k)
    data = {
        "chunks": chunks,
        "settings": settings,
        "timing": {"load_s": round(t1 - t0, 2), "transcribe_s": round(t2 - t1, 2)},
        "mlx_peak_memory_mb": round(mx.get_peak_memory() / 2**20, 1),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
