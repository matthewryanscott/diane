"""Diarizer candidate: Nemotron-3-Diarization through the mlx-audio port.

One process per file (model load included in runtime), run with the
eval/envs/nemotron-mlx interpreter directly so /usr/bin/time measures the model
process. The worker is nemotron_mlx_worker.py.

Outputs: runs/diar/<name>/<variant>/<meeting>.{rttm,meta.json,...}

Usage: uv run python run_nemotron_mlx.py [--name nemotron3-mlx]
       [--variants sdm,ihm] [--meetings all] [--preset offline]
"""

import argparse
import sys

from common import EVAL_DIR, audio_path, diar_run_dir, meetings_for, read_json, wav_duration, write_json
from measure import measure

ENV_PYTHON = EVAL_DIR / "envs" / "nemotron-mlx" / ".venv" / "bin" / "python"
WORKER = EVAL_DIR / "nemotron_mlx_worker.py"


def run_one(name: str, wav, file_id: str, out_dir, preset: str, force: bool) -> None:
    if (out_dir / f"{file_id}.meta.json").exists() and not force:
        return
    cmd = [str(ENV_PYTHON), str(WORKER), str(wav), file_id, str(out_dir), "--preset", preset]
    meta = measure(cmd, log=out_dir / f"{file_id}.log")
    stats = read_json(out_dir / f"{file_id}.nemotron.json")
    write_json(out_dir / f"{file_id}.meta.json", {
        "model": stats["model"],
        "settings": {
            "port": "mlx-audio",
            "preset": preset,
            "streaming": stats["streaming"],
            "dtype": stats["dtype"],
            "threshold": stats["threshold"],
            "min_duration_s": stats["min_duration_s"],
        },
        **meta,
        "audio_s": round(wav_duration(wav), 2),
        "nemotron": {k: stats[k] for k in ("load_s", "audio_load_s", "infer_s", "frames", "n_speakers")},
    })
    print(f"{name}/{out_dir.name}/{file_id}: {meta} infer_s={stats['infer_s']}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="nemotron3-mlx")
    parser.add_argument("--variants", default="sdm,ihm")
    parser.add_argument("--meetings", default="all")
    parser.add_argument("--preset", default="offline")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for variant in args.variants.split(","):
        for meeting in meetings_for(args.meetings):
            run_one(args.name, audio_path(variant, meeting), meeting, diar_run_dir(args.name, variant),
                    args.preset, args.force)


if __name__ == "__main__":
    main()
