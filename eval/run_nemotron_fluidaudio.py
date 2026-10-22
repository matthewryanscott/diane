"""Diarizer candidate: Nemotron-3-Diarization through FluidAudio's CoreML Swift CLI.

Runs `fluidaudiocli nemotron3-diarize <wav> --variant <v> --dump-preds ...`, one
process per file (model load included in runtime), measured directly. The CLI's
own RTTM uses a 0.2 s minimum segment; we keep it as <id>.cli.rttm and write
<id>.rttm from the dumped [T, 8] probabilities with nemotron_common's
binarization (threshold 0.5, no minimum duration), the same as the MLX port.
FluidAudio exposes no speaker embeddings.

Build once (eval/envs/fluidaudio is a FluidAudio checkout; see eval/results/nemotron3.md):
    cd envs/fluidaudio && swift build -c release --product fluidaudiocli

Usage: uv run python run_nemotron_fluidaudio.py [--name nemotron3-fluidaudio]
       [--variants sdm,ihm] [--meetings all] [--variant offline] [--compute-units all]
"""

import argparse
import re
import sys

import numpy as np

from common import EVAL_DIR, audio_path, diar_run_dir, meetings_for, wav_duration, write_json
from measure import measure
from nemotron_common import probs_to_segments, write_rttm

FA_DIR = EVAL_DIR / "envs" / "fluidaudio"
CLI = FA_DIR / ".build" / "release" / "fluidaudiocli"
_LOAD_RE = re.compile(r"Model loaded in ([\d.]+)s")
_PROC_RE = re.compile(r"Processed ([\d.]+)s in ([\d.]+)s")


def fa_commit() -> str:
    head = (FA_DIR / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        return (FA_DIR / ".git" / head[5:]).read_text().strip()[:10]
    return head[:10]


def run_one(name: str, wav, file_id: str, out_dir, variant: str, compute_units: str, force: bool) -> None:
    if (out_dir / f"{file_id}.meta.json").exists() and not force:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"{file_id}.preds.f32"
    log = out_dir / f"{file_id}.log"
    cmd = [str(CLI), "nemotron3-diarize", str(wav), "--variant", variant, "--compute-units", compute_units,
           "--dump-preds", str(preds_path), "--output", str(out_dir / f"{file_id}.cli.rttm")]
    meta = measure(cmd, log=log)
    probs = np.fromfile(preds_path, dtype="<f4").reshape(-1, 8)
    preds_path.unlink()
    np.save(out_dir / f"{file_id}.probs.npy", probs.astype(np.float16))
    segments = probs_to_segments(probs)
    write_rttm(out_dir / f"{file_id}.rttm", file_id, segments)
    text = log.read_text()
    load, proc = _LOAD_RE.search(text), _PROC_RE.search(text)
    write_json(out_dir / f"{file_id}.meta.json", {
        "model": f"FluidInference/nemotron-3-diarization-coreml:{variant}",
        "settings": {
            "port": "FluidAudio",
            "commit": fa_commit(),
            "variant": variant,
            "compute_units": compute_units,
            "threshold": 0.5,
            "min_duration_s": 0.0,
        },
        **meta,
        "audio_s": round(wav_duration(wav), 2),
        "nemotron": {
            "load_s": float(load.group(1)) if load else None,
            "infer_s": float(proc.group(2)) if proc else None,
            "frames": int(probs.shape[0]),
            "n_speakers": len({s for s, _, _ in segments}),
        },
    })
    print(f"{name}/{out_dir.name}/{file_id}: {meta} infer_s={proc.group(2) if proc else '?'}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="nemotron3-fluidaudio")
    parser.add_argument("--variants", default="sdm,ihm")
    parser.add_argument("--meetings", default="all")
    parser.add_argument("--variant", default="offline")
    parser.add_argument("--compute-units", default="all", choices=["all", "ane", "gpu", "cpu"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for variant in args.variants.split(","):
        for meeting in meetings_for(args.meetings):
            run_one(args.name, audio_path(variant, meeting), meeting, diar_run_dir(args.name, variant),
                    args.variant, args.compute_units, args.force)


if __name__ == "__main__":
    main()
