"""Find dropped regions: long hypothesis gaps where the reference has many words.

A gap is a stretch of >= --min-gap seconds with no hypothesis word (between two
hypothesis words, or before the first / after the last). A gap is a dropped region
when >= --min-words normalized reference words have their midpoint inside it.
Reference word times come from the AMI manual words.xml (real words only, the same
filter as prepare_ami.py); fillers that the Whisper normalizer removes don't count.

  uv run python analyze_gaps.py --names parakeet-v3,parakeet-v3-c60 [--variant both] [--list parakeet-v3]

Prints a summary table per variant (dropped regions, reference words inside them,
and how many regions touch a chunk-overlap zone, for fixed-chunk runs), then, with --list, every dropped region of the named runs with the number of
hypothesis words every other run put there. Writes runs/results/gaps__<variant>.json.
"""

import argparse
import bisect
import re
import warnings
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore", category=SyntaxWarning)  # whisper_normalizer docstring escapes

from common import MEETINGS, RESULTS_DIR, asr_run_dir, read_json, write_json
from prepare_ami import MANUAL_DIR, _speaker_names
from whisper_normalizer.english import EnglishTextNormalizer

_normalize = EnglishTextNormalizer()
CHUNK_RE = re.compile(r"--chunk-duration', '([\d.]+)'.*?--overlap-duration', '([\d.]+)'")


def ref_words(meeting: str) -> list[tuple[float, str]]:
    """(midpoint time, normalized word) for every real reference word, sorted by time."""
    words = []
    for agent in _speaker_names(meeting):
        path = MANUAL_DIR / "words" / f"{meeting}.{agent}.words.xml"
        if not path.exists():
            continue
        for el in ET.parse(path).getroot():
            if el.tag != "w" or el.get("punc") == "true" or el.get("starttime") is None:
                continue
            norm = " ".join(_normalize((el.text or "").strip()).split())
            if norm:
                end = float(el.get("endtime") or el.get("starttime"))
                words.append(((float(el.get("starttime")) + end) / 2, norm))
    return sorted(words)


def hyp_gaps(words: list[dict], duration: float, min_gap: float) -> list[tuple[float, float]]:
    """Stretches of >= min_gap seconds with no hypothesis word."""
    spans = sorted((w["start"], w["end"]) for w in words)
    gaps = []
    prev_end = 0.0
    for start, end in spans:
        if start - prev_end >= min_gap:
            gaps.append((prev_end, start))
        prev_end = max(prev_end, end)
    if duration - prev_end >= min_gap:
        gaps.append((prev_end, duration))
    return gaps


def chunk_overlaps(meta: dict, duration: float) -> list[tuple[float, float]]:
    """Overlap zones of parakeet-mlx fixed chunking, from the run's CLI args (default 120/15)."""
    settings = meta.get("settings", {})
    joined = str(settings.get("cli_args", ""))
    if "windows" in settings or "--chunk-duration', '0'" in joined:
        return []
    m = CHUNK_RE.search(joined)
    chunk, overlap = (float(m.group(1)), float(m.group(2))) if m else (120.0, 15.0)
    step = chunk - overlap
    return [(k * step, k * step + overlap) for k in range(1, int(duration // step) + 1)]


def count_in(times: list[float], start: float, end: float) -> int:
    return bisect.bisect_right(times, end) - bisect.bisect_left(times, start)


def analyze(name: str, variant: str, meeting: str, refs: list[tuple[float, str]], args) -> list[dict]:  # noqa: ANN001
    asr = read_json(asr_run_dir(name, variant) / f"{meeting}.json")
    duration = asr["meta"]["audio_s"]
    times = [t for t, _ in refs]
    zones = chunk_overlaps(asr["meta"], duration)
    regions = []
    for start, end in hyp_gaps(asr["words"], duration, args.min_gap):
        n = count_in(times, start, end)
        if n < args.min_words:
            continue
        lo, hi = bisect.bisect_left(times, start), bisect.bisect_right(times, end)
        near = min((max(z0 - end, start - z1, 0.0) for z0, z1 in zones), default=None)
        regions.append({
            "meeting": meeting,
            "start": round(start, 2),
            "end": round(end, 2),
            "ref_words": n,
            "near_overlap_s": None if near is None else round(near, 1),
            "ref_text": " ".join(w for _, w in refs[lo:hi])[:120],
        })
    return regions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--names", required=True, help="comma-separated ASR run names")
    parser.add_argument("--variant", default="both", choices=["sdm", "ihm", "both"])
    parser.add_argument("--min-gap", type=float, default=5.0)
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--list", default="", help="comma-separated run names whose regions to list")
    args = parser.parse_args()

    names = args.names.split(",")
    refs = {m: ref_words(m) for m in MEETINGS}
    total_ref = sum(len(r) for r in refs.values())
    for variant in ["sdm", "ihm"] if args.variant == "both" else [args.variant]:
        result = {n: [r for m in MEETINGS for r in analyze(n, variant, m, refs[m], args)] for n in names}
        print(f"\n### Dropped regions / {variant} (gap >= {args.min_gap:g} s, >= {args.min_words} ref words)\n")
        print("| run | regions | ref words in them | % of ref words | touching a chunk-overlap zone |")
        print("|---|---|---|---|---|")
        for n in names:
            regions = result[n]
            words = sum(r["ref_words"] for r in regions)
            chunked = any(r["near_overlap_s"] is not None for r in regions)
            near = [r for r in regions if r["near_overlap_s"] == 0]
            near_s = f"{len(near)} ({sum(r['ref_words'] for r in near)} words)" if chunked else "-"
            print(f"| {n} | {len(regions)} | {words} | {100 * words / total_ref:.2f} | {near_s} |")
        for n in filter(None, args.list.split(",")):
            others = [o for o in names if o != n]
            print(f"\n#### {n} / {variant}: dropped regions, and hyp words other runs put there\n")
            print("| meeting | start–end s | ref words | " + " | ".join(others) + " | reference (start) |")
            print("|---|---|---|" + "---|" * len(others) + "---|")
            for r in result[n]:
                cells = []
                for o in others:
                    hyp = read_json(asr_run_dir(o, variant) / f"{r['meeting']}.json")["words"]
                    cells.append(str(sum(1 for w in hyp if r["start"] <= (w["start"] + w["end"]) / 2 <= r["end"])))
                print(
                    f"| {r['meeting']} | {r['start']:.0f}–{r['end']:.0f} | {r['ref_words']} | "
                    + " | ".join(cells)
                    + f" | {r['ref_text'][:70]} |"
                )
        write_json(RESULTS_DIR / f"gaps__{variant}.json", {"args": vars(args), "total_ref_words": total_ref, **result})


if __name__ == "__main__":
    main()
