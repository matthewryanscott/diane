"""ASR candidate: Qwen3-ASR on MLX via mlx-qwen3-asr, word timestamps from Qwen3-ForcedAligner-0.6B.

The model runs in its own uv project (envs/qwen3/), whose interpreter is invoked
directly so /usr/bin/time measures the model process. mlx-qwen3-asr splits audio
at low-energy points into <=30 s chunks, decodes each greedily, and force-aligns
each chunk's text. Words keep the punctuation of the decoded text (as scribe's
parakeet words do); sentences split at . ? ! and at chunk boundaries.

Usage: uv run python run_qwen3.py [--name qwen3-1.7b] [--model Qwen/Qwen3-ASR-1.7B] [--dtype float16]
       [--variants sdm,ihm] [--meetings all] [--no-timestamps] [--force]
"""

import argparse
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

from common import EVAL_DIR, asr_run_dir, audio_path, meetings_for, read_json, wav_duration, write_json
from measure import measure

QWEN_ENV = EVAL_DIR / "envs" / "qwen3" / ".venv"
QWEN_PYTHON = QWEN_ENV / "bin" / "python"
WORKER = EVAL_DIR / "envs" / "qwen3" / "qwen3_worker.py"
ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"
SENTENCE_END = (".", "?", "!")


def _package_version(package: str) -> str:
    code = f"import importlib.metadata as m; print(m.version({package!r}))"
    return subprocess.run([QWEN_PYTHON, "-c", code], capture_output=True, text=True, check=True).stdout.strip()


def _clean(token: str) -> str:
    """mlx-qwen3-asr's ForcedAlignTextProcessor.clean_token: keep letters, digits and apostrophes."""
    return "".join(ch for ch in token if ch == "'" or unicodedata.category(ch)[0] in "LN")


def to_contract(raw: dict) -> tuple[dict, dict]:
    """Map the flat aligned-word list back onto each chunk's punctuated text.

    Returns the contract dict and alignment stats. A token the aligner did not return
    (its output ran short) gets a zero-length time at the previous word's end.
    """
    segments = raw.get("segments") or []
    counts = raw.get("aligned_per_chunk") or []
    chunks = [c for c in raw["chunks"] if c["text"].strip()]
    assert len(chunks) == len(counts), (len(chunks), len(counts))
    stats = {"tokens": 0, "unaligned": 0, "mismatched_text": 0}
    sentences: list[dict] = []
    pos = 0
    for chunk, count in zip(chunks, counts, strict=True):
        aligned = segments[pos : pos + count]
        pos += count
        k = 0
        current: list[dict] = []
        last_end = chunk["start"]

        def flush() -> None:
            if current:
                sentences.append({
                    "text": " ".join(w["text"] for w in current),
                    "start": current[0]["start"],
                    "end": current[-1]["end"],
                    "words": list(current),
                })
                current.clear()

        for token in chunk["text"].split():
            if not _clean(token):
                if current:  # stray punctuation: attach to the previous word
                    current[-1]["text"] += token
                continue
            stats["tokens"] += 1
            if k < len(aligned):
                a = aligned[k]
                if a["text"] != _clean(token):
                    stats["mismatched_text"] += 1
                start, end = a["start"], a["end"]
                k += 1
            else:
                stats["unaligned"] += 1
                start = end = last_end
            last_end = end
            current.append({"text": token, "start": round(start, 3), "end": round(end, 3)})
            if token.endswith(SENTENCE_END):
                flush()
        flush()
    assert pos == len(segments)
    return {"words": [w for s in sentences for w in s["words"]], "sentences": sentences}, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="qwen3-1.7b")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--variants", default="sdm,ihm")
    parser.add_argument("--meetings", default="all")
    parser.add_argument("--audio", type=Path, help="transcribe this file instead (writes <name>/custom/<stem>.json)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    settings = {
        "model_id": args.model,
        "aligner_id": ALIGNER,
        "dtype": args.dtype,
        "language": "English (forced)",
        "package": f"mlx-qwen3-asr=={_package_version('mlx-qwen3-asr')}",
        "mlx": _package_version("mlx"),
        "decoding": "greedy, energy-split chunks <=30 s, adaptive max_new_tokens",
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
            cmd = [str(QWEN_PYTHON), str(WORKER), str(wav), str(raw_path), "--model", args.model,
                   "--aligner", ALIGNER, "--dtype", args.dtype]  # fmt: skip
            meta = measure(cmd, log=out_dir / f"{meeting}.log")
            raw = read_json(raw_path)
        write_json(out_dir / f"{meeting}.raw.json", raw)
        result, stats = to_contract(raw)
        chunks = raw["chunks"]
        result["meta"] = {
            "model": args.name,
            "settings": settings,
            **meta,
            "audio_s": round(wav_duration(wav), 2),
            "worker_timing": raw["timing"],
            "mlx_peak_memory_mb": raw["mlx_peak_memory_mb"],
            "chunks": len(chunks),
            "truncated_chunks": sum(1 for c in chunks if c.get("truncated")),
            "alignment": stats,
        }
        write_json(out, result)
        print(f"{args.name}/{variant}/{meeting}: {meta} {raw['timing']} {stats}", file=sys.stderr)


if __name__ == "__main__":
    main()
