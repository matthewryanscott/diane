"""Worker for run_pyannote.py: run pyannote speaker-diarization-community-1 on one wav.

Runs in eval/envs/pyannote (pyannote.audio 4.x). One call writes both outputs of
the pipeline: the regular (overlap-aware) diarization to <out_dir>/<id>.rttm and
the exclusive one (at most one speaker at a time) to <excl_dir>/<id>.rttm. The
per-speaker embeddings (256-d, in speaker_diarization.labels() order) go to
<id>.embeddings.npy in both dirs, with the label order in <id>.speakers.json.
The wav is read with the stdlib (16 kHz mono s16) and passed as a waveform, so
torchcodec/ffmpeg is not involved. Per-step wall times come from the pipeline hook.

Usage: python pyannote_worker.py <wav> <id> <out_dir> <excl_dir> [--device mps|cpu]
       [--num-speakers N] [--min-speakers N] [--max-speakers N]
"""

import argparse
import json
import os
import time
import wave
from pathlib import Path

import numpy as np
import torch
from pyannote.audio import Pipeline

MODEL = "pyannote/speaker-diarization-community-1"


def load_wav(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2, path
        sr = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return torch.from_numpy(pcm.astype(np.float32) / 32768.0).unsqueeze(0), sr


class StepTimer:
    """Pipeline hook: wall time from the previous step's end to each step's last call."""

    def __init__(self) -> None:
        self.t_prev = time.perf_counter()
        self.last: dict[str, float] = {}
        self.order: list[str] = []

    def __call__(self, step_name, step_artefact, file=None, completed=None, total=None) -> None:
        if step_name not in self.last:
            self.order.append(step_name)
        self.last[step_name] = time.perf_counter()

    def stats(self) -> dict[str, float]:
        out, prev = {}, self.t_prev
        for name in self.order:
            out[f"{name}_s"] = round(self.last[name] - prev, 2)
            prev = self.last[name]
        return out


def write_outputs(annotation, labels, embeddings, file_id: str, out_dir: Path) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{file_id}.rttm").open("w") as fh:
        annotation.write_rttm(fh)
    durations = {str(s): round(float(annotation.label_duration(s)), 1) for s in labels}
    if embeddings is not None:
        np.save(out_dir / f"{file_id}.embeddings.npy", embeddings.astype(np.float32))
    (out_dir / f"{file_id}.speakers.json").write_text(json.dumps(
        {"labels": [str(s) for s in labels], "durations_s": durations,
         "embeddings": f"{file_id}.embeddings.npy" if embeddings is not None else None}, indent=1))
    return durations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    parser.add_argument("file_id")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("excl_dir", type=Path)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats: dict = {"model": MODEL, "device": args.device, "torch": torch.__version__,
                   "threads": torch.get_num_threads()}
    t0 = time.perf_counter()
    pipeline = Pipeline.from_pretrained(MODEL)
    pipeline.to(torch.device(args.device))
    stats["load_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    waveform, sr = load_wav(args.wav)
    stats["read_s"] = round(time.perf_counter() - t0, 2)
    hints = {k: v for k, v in (("num_speakers", args.num_speakers), ("min_speakers", args.min_speakers),
                               ("max_speakers", args.max_speakers)) if v is not None}
    stats["settings"] = {"hints": hints, "params": pipeline.parameters(instantiated=True),
                         "segmentation_batch_size": pipeline.segmentation_batch_size,
                         "embedding_batch_size": pipeline.embedding_batch_size}

    timer = StepTimer()
    t0 = time.perf_counter()
    output = pipeline({"waveform": waveform, "sample_rate": sr, "uri": args.file_id}, hook=timer, **hints)
    if args.device == "mps":
        torch.mps.synchronize()
    stats["infer_s"] = round(time.perf_counter() - t0, 2)
    stats["steps"] = timer.stats()

    labels = output.speaker_diarization.labels()
    emb = output.speaker_embeddings
    stats["embeddings_shape"] = list(emb.shape) if emb is not None else None
    stats["n_speakers"] = len(labels)
    stats["speaker_durations_s"] = write_outputs(output.speaker_diarization, labels, emb, args.file_id, args.out_dir)
    stats["exclusive_speaker_durations_s"] = write_outputs(
        output.exclusive_speaker_diarization, labels, emb, args.file_id, args.excl_dir)
    stats["n_speakers_exclusive"] = len(output.exclusive_speaker_diarization.labels())
    (args.out_dir / f"{args.file_id}.pyannote.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, default=str), flush=True)
    os._exit(0)  # skip slow interpreter teardown of torch/MPS state


if __name__ == "__main__":
    main()
