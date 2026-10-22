"""Shared paths, meeting lists, and file-format helpers for the AMI eval harness."""

import json
import sys
import wave
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_DIR = EVAL_DIR.parent

# `import scribe` anywhere in the harness means the working-tree scribe/scribe.py
# (stdlib-only at import time; senko is imported lazily by _create_diarizer).
sys.path.insert(0, str(REPO_DIR / "scribe"))
DATA_DIR = EVAL_DIR / "data" / "ami"
RUNS_DIR = EVAL_DIR / "runs"
RESULTS_DIR = RUNS_DIR / "results"

# AMI full-corpus-ASR test meetings (a subset of the 16-meeting test split).
MEETINGS = [
    "ES2004a",
    "ES2004b",
    "IS1009a",
    "IS1009b",
    "TS3003a",
    "TS3003b",
    "EN2002a",
    "EN2002b",
]
QUICK = [m for m in MEETINGS if m.endswith("a")]
SUBSETS = {"all": MEETINGS, "quick": QUICK}

# variant -> AMI mirror stream name
VARIANTS = {"sdm": "Array1-01", "ihm": "Mix-Headset"}


def meetings_for(subset_or_list: str) -> list[str]:
    """Resolve 'all', 'quick', or a comma-separated list of meeting ids."""
    if subset_or_list in SUBSETS:
        return SUBSETS[subset_or_list]
    return [m.strip() for m in subset_or_list.split(",") if m.strip()]


def audio_path(variant: str, meeting: str) -> Path:
    return DATA_DIR / "audio" / variant / f"{meeting}.wav"


def ref_seglst_path(meeting: str) -> Path:
    return DATA_DIR / "ref" / "seglst" / f"{meeting}.json"


def ref_rttm_path(meeting: str) -> Path:
    return DATA_DIR / "ref" / "rttm" / f"{meeting}.rttm"


def ref_uem_path(meeting: str) -> Path:
    return DATA_DIR / "ref" / "uem" / f"{meeting}.uem"


def asr_run_dir(name: str, variant: str) -> Path:
    return RUNS_DIR / "asr" / name / variant


def diar_run_dir(name: str, variant: str) -> Path:
    return RUNS_DIR / "diar" / name / variant


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def read_rttm(path: Path) -> list[tuple[str, float, float]]:
    """Return (speaker, start, end) triples from an RTTM file."""
    segments = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] != "SPEAKER":
            continue
        start, dur = float(fields[3]), float(fields[4])
        segments.append((fields[7], start, start + dur))
    return segments


def write_rttm(path: Path, meeting: str, segments: list[tuple[str, float, float]]) -> None:
    """Write (speaker, start, end) triples as RTTM."""
    lines = [
        f"SPEAKER {meeting} 1 {start:.3f} {end - start:.3f} <NA> <NA> {speaker} <NA> <NA>"
        for speaker, start, end in sorted(segments, key=lambda s: s[1])
        if end > start
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")
