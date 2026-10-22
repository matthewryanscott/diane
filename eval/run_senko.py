"""Diarizer candidate: senko exactly as scribe runs it (Diarizer(device="auto"), merged_segments).

One process per meeting (model load included in runtime), run with the main diane
venv's interpreter directly so /usr/bin/time measures the model process.

Usage: uv run python run_senko.py [--name senko] [--variants sdm,ihm] [--meetings all]
"""

import argparse
import sys

from common import EVAL_DIR, REPO_DIR, audio_path, diar_run_dir, meetings_for, read_json, wav_duration, write_json
from measure import measure

MAIN_PYTHON = REPO_DIR / ".venv" / "bin" / "python"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="senko")
    parser.add_argument("--variants", default="sdm,ihm")
    parser.add_argument("--meetings", default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for variant in args.variants.split(","):
        out_dir = diar_run_dir(args.name, variant)
        for meeting in meetings_for(args.meetings):
            if (out_dir / f"{meeting}.meta.json").exists() and not args.force:
                continue
            wav = audio_path(variant, meeting)
            cmd = [str(MAIN_PYTHON), str(EVAL_DIR / "senko_worker.py"), str(wav), meeting, str(out_dir)]
            meta = measure(cmd, log=out_dir / f"{meeting}.log")
            senko_stats = read_json(out_dir / f"{meeting}.senko.json")
            write_json(out_dir / f"{meeting}.meta.json", {
                "model": args.name,
                "settings": {"device": "auto", "via": "scribe._create_diarizer + scribe.run_diarization"},
                **meta,
                "audio_s": round(wav_duration(wav), 2),
                "senko": senko_stats,
            })
            print(f"{args.name}/{variant}/{meeting}: {meta}", file=sys.stderr)


if __name__ == "__main__":
    main()
