"""ASR candidate: IBM Granite-speech-4.1-2B on MLX via mlx-audio (text only, approximate word times).

The model runs in its own uv project (envs/granite-cohere/), whose interpreter is
invoked directly so /usr/bin/time measures the model process. The worker cuts the
audio at the quietest 100 ms window 20-30 s into each chunk and decodes each chunk
greedily with the raw-transcript prompt (lower case, no punctuation, closest to
verbatim). Granite emits no timestamps, so each chunk's words are spread evenly over
the chunk ("word_times": "approximate"); each chunk becomes one sentence.

run_cohere.py reuses this module's runner for Cohere Transcribe.

Usage: uv run python run_granite.py [--name granite-4.1-2b] [--model ibm-granite/granite-speech-4.1-2b]
       [--variants sdm,ihm] [--meetings all] [--audio FILE] [--force] [-- extra worker args]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from common import EVAL_DIR, asr_run_dir, audio_path, meetings_for, read_json, wav_duration, write_json
from measure import measure

GC_ENV = EVAL_DIR / "envs" / "granite-cohere" / ".venv"
GC_PYTHON = GC_ENV / "bin" / "python"
WORKER = EVAL_DIR / "envs" / "granite-cohere" / "gc_worker.py"
SENTENCE_END = (".", "?", "!")


def _package_version(package: str) -> str:
    code = f"import importlib.metadata as m; print(m.version({package!r}))"
    return subprocess.run([GC_PYTHON, "-c", code], capture_output=True, text=True, check=True).stdout.strip()


def to_contract(raw: dict) -> dict:
    """Spread each chunk's words evenly over the chunk; split sentences at . ? ! and at chunk ends."""
    sentences: list[dict] = []
    for chunk in raw["chunks"]:
        tokens = chunk["text"].split()
        if not tokens:
            continue
        step = (chunk["end"] - chunk["start"]) / len(tokens)
        current: list[dict] = []
        for i, token in enumerate(tokens):
            start = chunk["start"] + i * step
            current.append({"text": token, "start": round(start, 3), "end": round(start + step, 3)})
            if token.endswith(SENTENCE_END) or i == len(tokens) - 1:
                sentences.append({
                    "text": " ".join(w["text"] for w in current),
                    "start": current[0]["start"],
                    "end": current[-1]["end"],
                    "words": current,
                })
                current = []
    return {"words": [w for s in sentences for w in s["words"]], "sentences": sentences}


def run(kind: str, default_name: str, default_model: str, doc: str) -> None:
    parser = argparse.ArgumentParser(description=doc.splitlines()[0])
    parser.add_argument("--name", default=default_name)
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--variants", default="sdm,ihm")
    parser.add_argument("--meetings", default="all")
    parser.add_argument("--audio", type=Path, help="transcribe this file instead (writes <name>/custom/<stem>.json)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("extra", nargs="*", help="extra gc_worker.py args, after --")
    args = parser.parse_args()

    settings = {
        "model_id": args.model,
        "package": f"mlx-audio=={_package_version('mlx-audio')}",
        "mlx": _package_version("mlx"),
        "worker_args": args.extra,
    }
    jobs = (
        [("custom", args.audio.stem, args.audio)]
        if args.audio
        else [(v, m, audio_path(v, m)) for v in args.variants.split(",") for m in meetings_for(args.meetings)]
    )
    for variant, meeting, wav in jobs:
        out_dir = asr_run_dir(args.name, variant)
        out = out_dir / f"{meeting}.json"
        if out.exists() and not args.force:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "raw.json"
            cmd = [str(GC_PYTHON), str(WORKER), kind, str(wav), str(raw_path), "--model", args.model, *args.extra]
            meta = measure(cmd, log=out_dir / f"{meeting}.log")
            raw = read_json(raw_path)
        write_json(out_dir / f"{meeting}.raw.json", raw)
        result = to_contract(raw)
        result["meta"] = {
            "model": args.name,
            "settings": {**settings, **raw["settings"]},
            **meta,
            "audio_s": round(wav_duration(wav), 2),
            "word_times": "approximate",
            "worker_timing": raw["timing"],
            "mlx_peak_memory_mb": raw["mlx_peak_memory_mb"],
            "chunks": len(raw["chunks"]),
        }
        write_json(out, result)
        print(f"{args.name}/{variant}/{meeting}: {meta} {raw['timing']} words={len(result['words'])}", file=sys.stderr)


if __name__ == "__main__":
    run("granite", "granite-4.1-2b", "ibm-granite/granite-speech-4.1-2b", __doc__)
