"""Download AMI test meetings and build gold references.

Produces, under eval/data/ami/:
  audio/{sdm,ihm}/<M>.wav        16 kHz mono wav (SDM = Array1-01, IHM = Mix-Headset)
  ref/seglst/<M>.json            meeteval SegLST, one segment per AMI manual segment
  ref/rttm/<M>.rttm              pyannote AMI-diarization-setup "only_words" RTTM (for DER)
  ref/uem/<M>.uem                pyannote AMI-diarization-setup UEM (scored region)
  ref/rttm_from_words/<M>.rttm   RTTM derived from our SegLST (cross-check only)
  meetings.json                  duration, speaker count, word count per meeting

Usage: uv run python prepare_ami.py [--meetings all|quick|M1,M2]
"""

import argparse
import re
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from common import (
    DATA_DIR,
    QUICK,
    VARIANTS,
    audio_path,
    meetings_for,
    ref_rttm_path,
    ref_seglst_path,
    ref_uem_path,
    write_json,
    write_rttm,
)

AMI_AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.{stream}.wav"
AMI_MANUAL_URL = "http://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
PYANNOTE_URL = "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main"
NITE = "{http://nite.sourceforge.net/}"
RAW_DIR = DATA_DIR / "raw"
MANUAL_DIR = RAW_DIR / "manual"
HREF_RE = re.compile(r"id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}", file=sys.stderr)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310
    tmp.rename(dest)


def prepare_audio(meeting: str) -> None:
    for variant, stream in VARIANTS.items():
        out = audio_path(variant, meeting)
        if out.exists():
            continue
        raw = RAW_DIR / "audio" / f"{meeting}.{stream}.wav"
        _download(AMI_AUDIO_URL.format(m=meeting, stream=stream), raw)
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(raw), "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", str(out), "-y"],
            check=True,
        )  # fmt: skip


def _speaker_names(meeting: str) -> dict[str, str]:
    """Map NXT agent letter (A-E) to AMI global speaker id (e.g. MEO015)."""
    root = ET.parse(MANUAL_DIR / "corpusResources" / "meetings.xml").getroot()
    for m in root.iter("meeting"):
        if m.get("observation") == meeting:
            return {s.get("nxt_agent"): s.get("global_name") for s in m.iter("speaker")}
    msg = f"{meeting} not in meetings.xml"
    raise KeyError(msg)


def _segments_for_agent(meeting: str, agent: str, speaker: str) -> list[dict]:
    words_root = ET.parse(MANUAL_DIR / "words" / f"{meeting}.{agent}.words.xml").getroot()
    elements = list(words_root)
    index = {el.get(f"{NITE}id"): i for i, el in enumerate(elements)}
    segs_root = ET.parse(MANUAL_DIR / "segments" / f"{meeting}.{agent}.segments.xml").getroot()
    segments = []
    for seg in segs_root.iter("segment"):
        words: list[str] = []
        starts: list[float] = []
        ends: list[float] = []
        for child in seg.iter(f"{NITE}child"):
            match = HREF_RE.search(child.get("href", ""))
            if not match:
                continue
            first = index[match.group(1)]
            last = index[match.group(2) or match.group(1)]
            for el in elements[first : last + 1]:
                # Only real words count: punctuation, vocal sounds, disfluency markers,
                # and gaps are dropped (the "only_words" convention).
                if el.tag != "w" or el.get("punc") == "true" or not (el.text or "").strip():
                    continue
                words.append(el.text.strip())
                if el.get("starttime") is not None:
                    starts.append(float(el.get("starttime")))
                if el.get("endtime") is not None:
                    ends.append(float(el.get("endtime")))
        if not words:
            continue
        start = min(starts) if starts else float(seg.get("transcriber_start"))
        end = max(ends) if ends else float(seg.get("transcriber_end"))
        segments.append({
            "session_id": meeting,
            "speaker": speaker,
            "start_time": round(start, 3),
            "end_time": round(max(end, start), 3),
            "words": " ".join(words),
        })
    return segments


def prepare_refs(meeting: str) -> dict:
    _download(f"{PYANNOTE_URL}/only_words/rttms/test/{meeting}.rttm", ref_rttm_path(meeting))
    _download(f"{PYANNOTE_URL}/uems/test/{meeting}.uem", ref_uem_path(meeting))

    segments = []
    for agent, speaker in sorted(_speaker_names(meeting).items()):
        if (MANUAL_DIR / "words" / f"{meeting}.{agent}.words.xml").exists():
            segments.extend(_segments_for_agent(meeting, agent, speaker))
    segments.sort(key=lambda s: (s["start_time"], s["speaker"]))
    write_json(ref_seglst_path(meeting), segments)
    write_rttm(
        DATA_DIR / "ref" / "rttm_from_words" / f"{meeting}.rttm",
        meeting,
        [(s["speaker"], s["start_time"], s["end_time"]) for s in segments],
    )

    uem_fields = ref_uem_path(meeting).read_text().split()
    duration = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
             str(audio_path("ihm", meeting))],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    )  # fmt: skip
    return {
        "duration_s": round(duration, 1),
        "uem": [float(uem_fields[2]), float(uem_fields[3])],
        "n_speakers": len({s["speaker"] for s in segments}),
        "n_segments": len(segments),
        "n_words": sum(len(s["words"].split()) for s in segments),
        "quick": meeting in QUICK,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--meetings", default="all")
    args = parser.parse_args()

    zip_path = RAW_DIR / "ami_public_manual_1.6.2.zip"
    _download(AMI_MANUAL_URL, zip_path)
    if not (MANUAL_DIR / "words").exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(MANUAL_DIR)

    info_path = DATA_DIR / "meetings.json"
    info = {}
    for meeting in meetings_for(args.meetings):
        prepare_audio(meeting)
        info[meeting] = prepare_refs(meeting)
        print(meeting, info[meeting], file=sys.stderr)
    write_json(info_path, info)


if __name__ == "__main__":
    main()
