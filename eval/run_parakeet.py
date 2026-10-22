"""ASR candidate: parakeet-mlx exactly as scribe runs it (default model, 120 s chunks, 15 s overlap).

Runs the parakeet-mlx executable from the uvx environment directly (not through
uvx) so /usr/bin/time sees the model process. Words are grouped from subword
tokens with scribe's own _words_from_tokens.

Usage: uv run python run_parakeet.py [--name parakeet-v3] [--variants sdm,ihm] [--meetings all]
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from common import asr_run_dir, audio_path, meetings_for, read_json, wav_duration, write_json
from measure import measure

import scribe  # resolved via the sys.path entry common adds


def _parakeet_env() -> Path:
    """Return the uvx-managed environment scribe's parakeet-mlx spec resolves to."""
    out = subprocess.run(
        ["uvx", "--from", scribe._PARAKEET_VERSION, "python", "-c", "import sys; print(sys.prefix)"],  # noqa: SLF001
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return Path(out.stdout.strip())


def _package_version(env: Path, package: str) -> str:
    code = f"import importlib.metadata as m; print(m.version({package!r}))"
    return subprocess.run(
        [env / "bin" / "python", "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()


def to_contract(data: dict) -> dict:
    """Convert parakeet-mlx JSON (sentences[].tokens[]) to the ASR contract."""
    sentences = []
    for sentence in data.get("sentences", []):
        text = sentence.get("text", "").strip()
        if not text:
            continue
        words = [
            {"text": w.text, "start": round(w.start, 3), "end": round(w.end, 3)}
            for w in scribe._words_from_tokens(sentence.get("tokens", []))  # noqa: SLF001
        ]
        sentences.append({
            "text": text,
            "start": round(float(sentence["start"]), 3),
            "end": round(float(sentence["end"]), 3),
            "words": words,
        })
    return {"words": [w for s in sentences for w in s["words"]], "sentences": sentences}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="parakeet-v3")
    parser.add_argument("--variants", default="sdm,ihm")
    parser.add_argument("--meetings", default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("extra", nargs="*", help="extra parakeet-mlx CLI args (e.g. --model X)")
    args = parser.parse_args()

    env = _parakeet_env()
    exe = env / "bin" / "parakeet-mlx"
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("PARAKEET_")}
    settings = {
        "model_id": "mlx-community/parakeet-tdt-0.6b-v3 (parakeet-mlx default)",
        "package": f"parakeet-mlx=={_package_version(env, 'parakeet-mlx')}",
        "mlx": _package_version(env, "mlx"),
        "cli_args": args.extra or "defaults (model, chunk 120 s, overlap 15 s, greedy, bf16)",
    }
    for variant in args.variants.split(","):
        out_dir = asr_run_dir(args.name, variant)
        for meeting in meetings_for(args.meetings):
            out = out_dir / f"{meeting}.json"
            if out.exists() and not args.force:
                continue
            wav = audio_path(variant, meeting)
            with tempfile.TemporaryDirectory() as tmp:
                cmd = [str(exe), str(wav), "--output-format", "json", "--output-dir", tmp, *args.extra]
                meta = measure(cmd, log=out_dir / f"{meeting}.log", env=clean_env)
                raw = read_json(Path(tmp) / f"{meeting}.json")
                write_json(out_dir / f"{meeting}.raw.json", raw)  # keeps tokens for regrouping
                result = to_contract(raw)
            result["meta"] = {
                "model": args.name,
                "settings": settings,
                **meta,
                "audio_s": round(wav_duration(wav), 2),
            }
            write_json(out, result)
            print(f"{args.name}/{variant}/{meeting}: {meta}", file=sys.stderr)


if __name__ == "__main__":
    main()
