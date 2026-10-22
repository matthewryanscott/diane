"""Timestamp, disfluency and hallucination diagnostics for ASR runs (written for the Qwen3-ASR experiment).

For each ASR run and variant it reports:
- word timing shape: median/p90 duration, share of zero-length words;
- word timing accuracy against the AMI manual word times (ami_public_manual_1.6.2 words/*.xml):
  hypothesis and reference word streams are matched with difflib on lowercased tokens, and
  |start error| / |end error| are summarized over exactly matched words;
- words far from any reference word (midpoint > 1 s from every reference word), with their text;
- repetition loops (an n-gram, n = 1..6, repeated back-to-back >= 4 times) and stretches where the
  hypothesis has far more words than the reference in a 30 s window;
- disfluency style: fillers per 1000 words and immediate word repetitions ("the the") per 1000 words,
  compared with the reference.

Usage: uv run python analyze_qwen3.py --names parakeet-v3,qwen3-1.7b [--variants sdm,ihm] [--meetings all]
"""

import argparse
import bisect
import difflib
import re
import statistics
import zipfile
from functools import cache
from xml.etree import ElementTree

from common import DATA_DIR, RESULTS_DIR, asr_run_dir, meetings_for, read_json, write_json

FILLERS = {"um", "uh", "hmm", "mm", "er", "ah", "erm", "uhm", "mhm", "mm-hmm", "uh-huh", "hm"}
FAR_S = 1.0
_TOKEN_RE = re.compile(r"[^a-z0-9'\-]")


def tok(text: str) -> str:
    return _TOKEN_RE.sub("", text.lower()).strip("-'")


@cache
def ref_words(meeting: str) -> list[tuple[str, float, float]]:
    """All speakers' manual words (no punctuation, vocal sounds or gaps), sorted by start time."""
    words = []
    zpath = DATA_DIR / "raw" / "ami_public_manual_1.6.2.zip"
    with zipfile.ZipFile(zpath) as zf:
        for name in zf.namelist():
            if not (name.startswith("words/") and name.split("/")[-1].startswith(meeting + ".")):
                continue
            root = ElementTree.fromstring(zf.read(name))
            for el in root:
                if el.tag != "w" or el.get("punc") == "true" or el.get("starttime") is None:
                    continue
                t = tok(el.text or "")
                if t:
                    words.append((t, float(el.get("starttime")), float(el.get("endtime"))))
    return sorted(words, key=lambda w: w[1])


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def quantile(xs: list[float], q: float) -> float:
    return statistics.quantiles(xs, n=100)[int(q * 100) - 1] if len(xs) > 1 else (xs[0] if xs else float("nan"))


def loops(tokens: list[str], min_repeats: int = 4, max_n: int = 6) -> list[tuple[int, int, int, str]]:
    """(index, n, repeats, text) for back-to-back repeats of an n-gram."""
    found = []
    i = 0
    while i < len(tokens):
        best = None
        for n in range(1, max_n + 1):
            gram = tokens[i : i + n]
            if len(gram) < n:
                break
            r = 1
            while tokens[i + r * n : i + (r + 1) * n] == gram:
                r += 1
            if r >= min_repeats and (best is None or r * n > best[1] * best[2]):
                best = (i, n, r, " ".join(gram))
        if best:
            found.append(best)
            i += best[1] * best[2]
        else:
            i += 1
    return found


def immediate_repeats(tokens: list[str]) -> int:
    return sum(1 for a, b in zip(tokens, tokens[1:]) if a == b and a not in FILLERS)


def analyze(name: str, variant: str, meeting: str) -> dict:
    asr = read_json(asr_run_dir(name, variant) / f"{meeting}.json")
    hyp = [(tok(w["text"]), w["start"], w["end"]) for w in asr["words"]]
    hyp = [w for w in hyp if w[0]]
    ref = ref_words(meeting)
    durations = [e - s for _, s, e in hyp]

    # timing accuracy on exactly matched words
    sm = difflib.SequenceMatcher(a=[w[0] for w in ref], b=[w[0] for w in hyp], autojunk=False)
    d_start, d_end = [], []
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            r, h = ref[block.a + k], hyp[block.b + k]
            d_start.append(abs(h[1] - r[1]))
            d_end.append(abs(h[2] - r[2]))

    # words far from any reference word
    ref_mids = sorted((s + e) / 2 for _, s, e in ref)
    ref_spans = sorted((s, e) for _, s, e in ref)
    starts = [s for s, _ in ref_spans]
    max_end_before = []
    m = float("-inf")
    for _, e in ref_spans:
        m = max(m, e)
        max_end_before.append(m)

    def far(t: float) -> bool:
        i = bisect.bisect_right(starts, t + FAR_S)
        if i == 0:
            return True
        return max_end_before[i - 1] < t - FAR_S

    far_words = [(w, (s + e) / 2) for w, s, e in hyp if far((s + e) / 2)]
    far_runs: list[tuple[float, str]] = []
    for w, t in far_words:
        if far_runs and t - far_runs[-1][0] < 5:
            far_runs[-1] = (t, far_runs[-1][1] + " " + w)
        else:
            far_runs.append((t, w))

    # 30 s windows where the hypothesis has many more words than the reference
    dense = []
    end_t = max([e for _, _, e in hyp] + [e for _, _, e in ref] + [0])
    for w0 in range(0, int(end_t) + 1, 15):
        nh = sum(1 for _, s, _ in hyp if w0 <= s < w0 + 30)
        nr = sum(1 for _, s, _ in ref if w0 <= s < w0 + 30)
        if nh - nr >= 25:
            dense.append((w0, nh, nr))

    htoks = [w[0] for w in hyp]
    rtoks_by_seg = [s["words"].lower().split() for s in read_json(DATA_DIR / "ref" / "seglst" / f"{meeting}.json")]
    rtoks = [tok(t) for seg in rtoks_by_seg for t in seg]
    return {
        "n_words": len(hyp),
        "durations": durations,
        "d_start": d_start,
        "d_end": d_end,
        "matched": len(d_start),
        "far_words": len(far_words),
        "far_runs": [(round(t, 1), text) for t, text in far_runs],
        "dense_windows": dense,
        "loops": [(round(hyp[i][1], 1), n, r, g) for i, n, r, g in loops(htoks)],
        "ref_loops": len(loops(rtoks)),
        "fillers": sum(1 for t in htoks if t in FILLERS),
        "ref_fillers": sum(1 for t in rtoks if t in FILLERS),
        "repeats": immediate_repeats(htoks),
        "ref_repeats": sum(immediate_repeats([tok(t) for t in seg]) for seg in rtoks_by_seg),
        "ref_n": len(rtoks),
        "truncated_chunks": asr["meta"].get("truncated_chunks"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--names", required=True)
    parser.add_argument("--variants", default="sdm,ihm")
    parser.add_argument("--meetings", default="all")
    parser.add_argument("--show", type=int, default=6, help="far-word runs and loops to print per run")
    args = parser.parse_args()
    meetings = meetings_for(args.meetings)
    out = {}
    rows_t, rows_s, rows_h = [], [], []
    for name in args.names.split(","):
        for variant in args.variants.split(","):
            per = {m: analyze(name, variant, m) for m in meetings}
            dur = [d for p in per.values() for d in p["durations"]]
            ds = [d for p in per.values() for d in p["d_start"]]
            de = [d for p in per.values() for d in p["d_end"]]
            n = sum(p["n_words"] for p in per.values())
            ref_n = sum(p["ref_n"] for p in per.values())
            summary = {
                "n_words": n,
                "median_dur_s": statistics.median(dur),
                "p90_dur_s": quantile(dur, 0.9),
                "zero_len_frac": sum(1 for d in dur if d <= 0) / n,
                "long_frac": sum(1 for d in dur if d > 1.5) / n,
                "matched_frac": len(ds) / n,
                "start_err_median": statistics.median(ds),
                "start_err_p90": quantile(ds, 0.9),
                "start_within_0.2": sum(1 for d in ds if d <= 0.2) / len(ds),
                "end_err_median": statistics.median(de),
                "end_within_0.2": sum(1 for d in de if d <= 0.2) / len(de),
                "far_words": sum(p["far_words"] for p in per.values()),
                "loops": sum(len(p["loops"]) for p in per.values()),
                "ref_loops": sum(p["ref_loops"] for p in per.values()),
                "dense_windows": sum(len(p["dense_windows"]) for p in per.values()),
                "fillers_per_k": 1000 * sum(p["fillers"] for p in per.values()) / n,
                "ref_fillers_per_k": 1000 * sum(p["ref_fillers"] for p in per.values()) / ref_n,
                "repeats_per_k": 1000 * sum(p["repeats"] for p in per.values()) / n,
                "ref_repeats_per_k": 1000 * sum(p["ref_repeats"] for p in per.values()) / ref_n,
            }
            out[f"{name}/{variant}"] = {
                "summary": summary,
                "per_meeting": {m: {k: v for k, v in p.items() if k not in ("durations", "d_start", "d_end")} for m, p in per.items()},
            }
            s = summary
            rows_t.append([name, variant, n, f"{s['median_dur_s']:.2f}", f"{s['p90_dur_s']:.2f}", pct(s["zero_len_frac"]),
                           pct(s["long_frac"]), pct(s["matched_frac"]), f"{s['start_err_median']:.3f}",
                           f"{s['start_err_p90']:.2f}", pct(s["start_within_0.2"]), f"{s['end_err_median']:.3f}",
                           pct(s["end_within_0.2"])])  # fmt: skip
            rows_s.append([name, variant, f"{s['fillers_per_k']:.1f} (ref {s['ref_fillers_per_k']:.1f})",
                           f"{s['repeats_per_k']:.1f} (ref {s['ref_repeats_per_k']:.1f})"])  # fmt: skip
            rows_h.append([name, variant, s["far_words"], s["loops"], s["ref_loops"], s["dense_windows"]])
            for m, p in per.items():
                if p["loops"]:
                    print(f"  loops {name}/{variant}/{m}: {p['loops'][: args.show]}")
                if p["dense_windows"]:
                    print(f"  dense {name}/{variant}/{m}: {p['dense_windows'][: args.show]}")
                big = sorted(p["far_runs"], key=lambda r: -len(r[1]))[: args.show]
                if big:
                    print(f"  far {name}/{variant}/{m}: {big}")

    def table(header: list[str], rows: list[list]) -> str:
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])

    print("\n### Word timings\n")
    print(table(["run", "variant", "words", "median dur s", "p90 dur s", "zero-len %", ">1.5 s %", "matched %",
                 "median |start err| s", "p90 |start err| s", "start <=0.2 s %", "median |end err| s", "end <=0.2 s %"], rows_t))  # fmt: skip
    print("\n### Style\n")
    print(table(["run", "variant", "fillers /1k words", "immediate repeats /1k words"], rows_s))
    print("\n### Hallucination signals\n")
    print(table(["run", "variant", f"words >{FAR_S:.0f} s from any ref word", "n-gram loops (hyp)", "loops (ref)",
                 "30 s windows with >=25 extra words"], rows_h))  # fmt: skip
    write_json(RESULTS_DIR / f"qwen3_analysis__{args.names.replace(',', '+')}__{args.meetings}.json", out)


if __name__ == "__main__":
    main()
