"""Worker: diarize one wav with the mlx-audio port of Nemotron-3-Diarization.

Runs in eval/envs/nemotron-mlx (driven by run_nemotron_mlx.py). Writes into
out_dir:

- <id>.rttm           speakers speaker_0..7 in arrival order (threshold 0.5, no min duration)
- <id>.probs.npy      [T, 8] float16 activity probabilities at 10 ms
- <id>.centroids.npy  [n_active, 512] probability-weighted mean of the encoder's
                      final hidden state per active speaker (80 ms frames). The port
                      exposes no speaker embeddings; these are a by-product of the
                      diarizer's encoder, not identity embeddings.
- <id>.speakers.json  the speaker labels, in centroid row order
- <id>.nemotron.json  timing and settings

Usage: python nemotron_mlx_worker.py <wav> <id> <out_dir> [--preset offline] [--model ...]
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.vad import load

from nemotron_common import THRESHOLD, probs_to_segments, write_rttm

MODEL_ID = "mlx-community/Nemotron-3-Diarization"


def attach_centroid_recorder(model) -> dict:
    """Accumulate probability-weighted encoder states per speaker during streaming.

    streaming_step runs the encoder over [spkcache | fifo | chunk]; the chunk's
    central frames start at cache_len + fifo_len. We record the encoder output
    and pool it with the chunk's 80 ms probabilities.
    """
    acc = {"sum": None, "weight": None, "last_enc": None}
    encoder_cls = type(model.encoder)
    orig_encoder_call = encoder_cls.__call__

    def recording_call(self, x, lengths):
        out = orig_encoder_call(self, x, lengths)
        acc["last_enc"] = out
        return out

    encoder_cls.__call__ = recording_call
    orig_step = model.streaming_step
    factor = model.config.encoder_config.subsampling_factor

    def step(features, state, central_frames, feature_length=None):
        start = state.spkcache.shape[1] + state.fifo.shape[1]
        result = orig_step(features, state, central_frames, feature_length)
        n = (central_frames + factor - 1) // factor
        enc = np.asarray(acc["last_enc"][0, start : start + n].astype(mx.float32))
        probs = np.asarray(result)
        probs = np.pad(probs, [(0, n * factor - probs.shape[0]), (0, 0)]).reshape(n, factor, -1).mean(axis=1)
        enc = enc[: probs.shape[0]]
        weighted = probs.T @ enc
        if acc["sum"] is None:
            acc["sum"], acc["weight"] = weighted, probs.sum(axis=0)
        else:
            acc["sum"] += weighted
            acc["weight"] += probs.sum(axis=0)
        return result

    model.streaming_step = step
    return acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    parser.add_argument("file_id")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--preset", default="offline", choices=["offline", "low", "very_low", "ultra_low"])
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model = load(args.model, strict=True)
    model.set_streaming_config(args.preset)
    acc = attach_centroid_recorder(model)
    t_load = time.perf_counter()
    waveform, _ = model._load_audio(str(args.wav), 16000)
    mx.eval(waveform)
    t_audio = time.perf_counter()
    chunks = [np.asarray(r.speaker_probs) for r in model.generate_stream(waveform, 16000, threshold=args.threshold)]
    probs = np.concatenate(chunks) if chunks else np.zeros((0, model.config.num_speakers), np.float32)
    t_infer = time.perf_counter()

    segments = probs_to_segments(probs, threshold=args.threshold)
    write_rttm(args.out_dir / f"{args.file_id}.rttm", args.file_id, segments)
    np.save(args.out_dir / f"{args.file_id}.probs.npy", probs.astype(np.float16))
    active = sorted({s for s, _, _ in segments}, key=lambda s: int(s.split("_")[1]))
    if acc["sum"] is not None and active:
        idx = [int(s.split("_")[1]) for s in active]
        centroids = acc["sum"][idx] / np.maximum(acc["weight"][idx], 1e-6)[:, None]
        np.save(args.out_dir / f"{args.file_id}.centroids.npy", centroids.astype(np.float32))
    (args.out_dir / f"{args.file_id}.speakers.json").write_text(json.dumps(active))
    cfg = model.config.modules_config
    stats = {
        "model": args.model,
        "preset": args.preset,
        "streaming": {
            "chunk_len": cfg.chunk_len,
            "chunk_right_context": cfg.chunk_right_context,
            "fifo_len": cfg.fifo_len,
            "spkcache_len": cfg.spkcache_len,
            "spkcache_update_period": cfg.spkcache_update_period,
        },
        "dtype": str(model.dtype),
        "threshold": args.threshold,
        "min_duration_s": 0.0,
        "load_s": round(t_load - t0, 3),
        "audio_load_s": round(t_audio - t_load, 3),
        "infer_s": round(t_infer - t_audio, 3),
        "frames": int(probs.shape[0]),
        "n_speakers": len(active),
        "speaker_seconds": {s: round(sum(e - b for sp, b, e in segments if sp == s), 2) for s in active},
    }
    (args.out_dir / f"{args.file_id}.nemotron.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
