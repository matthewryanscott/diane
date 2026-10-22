"""Diarizer candidate: pyannote/speaker-diarization-community-1 (pyannote.audio 4.x).

One process per meeting (model load included in runtime), run with the
eval/envs/pyannote interpreter directly so /usr/bin/time measures the model
process. The worker is pyannote_worker.py. Each run writes two diarizer names:
<name> (the regular, overlap-aware output) and <name>-excl (the pipeline's
exclusive_speaker_diarization, at most one speaker at a time). Both get the same
meta.json, and both hold <M>.embeddings.npy (one 256-d embedding per speaker).

--num-speakers oracle gives each AMI meeting its true count from the reference RTTM.

Usage: uv run python run_pyannote.py [--name pyannote-c1] [--variants sdm,ihm] [--meetings all]
       [--device mps|cpu] [--num-speakers N|oracle] [--min-speakers N] [--max-speakers N]
"""

import argparse
import sys

from common import EVAL_DIR, audio_path, diar_run_dir, meetings_for, read_json, read_rttm, ref_rttm_path, wav_duration, write_json
from measure import measure

ENV_PYTHON = EVAL_DIR / "envs" / "pyannote" / ".venv" / "bin" / "python"
WORKER = EVAL_DIR / "pyannote_worker.py"


def run_one(args, wav, file_id: str, variant: str, num_speakers: int | None) -> None:
    out_dir, excl_dir = diar_run_dir(args.name, variant), diar_run_dir(f"{args.name}-excl", variant)
    if (out_dir / f"{file_id}.meta.json").exists() and not args.force:
        return
    cmd = [str(ENV_PYTHON), str(WORKER), str(wav), file_id, str(out_dir), str(excl_dir), "--device", args.device]
    if num_speakers is not None:
        cmd += ["--num-speakers", str(num_speakers)]
    if args.min_speakers is not None:
        cmd += ["--min-speakers", str(args.min_speakers)]
    if args.max_speakers is not None:
        cmd += ["--max-speakers", str(args.max_speakers)]
    meta = measure(cmd, log=out_dir / f"{file_id}.log")
    stats = read_json(out_dir / f"{file_id}.pyannote.json")
    for output, d in (("speaker_diarization", out_dir), ("exclusive_speaker_diarization", excl_dir)):
        write_json(d / f"{file_id}.meta.json", {
            "model": stats["model"],
            "settings": {"output": output, "device": stats["device"], "torch": stats["torch"],
                         "threads": stats["threads"], **stats["settings"]},
            **meta,
            "audio_s": round(wav_duration(wav), 2),
            "pyannote": {k: v for k, v in stats.items() if k.endswith("_s") or k in ("steps", "n_speakers")},
        })
    print(f"{args.name}/{variant}/{file_id}: {meta} infer_s={stats['infer_s']} "
          f"n_speakers={stats['n_speakers']}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="pyannote-c1")
    parser.add_argument("--variants", default="sdm,ihm")
    parser.add_argument("--meetings", default="all")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--num-speakers", default=None, help="an int, or 'oracle' (AMI reference count)")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for variant in args.variants.split(","):
        for meeting in meetings_for(args.meetings):
            n = args.num_speakers
            if n == "oracle":
                n = len({s for s, _, _ in read_rttm(ref_rttm_path(meeting))})
            run_one(args, audio_path(variant, meeting), meeting, variant, int(n) if n else None)


if __name__ == "__main__":
    main()
