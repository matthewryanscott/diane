"""Merge refinements: score word-grouping and speaker-smoothing variants offline.

Re-groups parakeet's saved tokens (`<meeting>.raw.json`) and re-merges them with a
diarizer's saved RTTM, so no model runs. Variants:

- words: `current` (scribe's _words_from_tokens) or `trim` (a word ends at its last
  non-punctuation token, and a punctuation-only token never starts a word).
- tiny: drop diarizer speakers with less than this share of total speech first.
- smooth: `none`; `island<k>` (a run of <= k words inside a sentence, between two runs
  of the same speaker, takes that speaker); `edge<k>` (also a run of <= k words at a
  sentence edge takes its neighbour's speaker unless a token gap >= 150 ms separates
  them).

  uv run python merge_refine.py [--asr parakeet-v3-c30] [--diar scribe-e2e] [--variants sdm,ihm]
"""

import argparse
import itertools
import logging
import string

import meeteval
from common import asr_run_dir, diar_run_dir, meetings_for, read_json, read_rttm
from meeteval.io import SegLST
from meeteval.wer import combine_error_rates
from score import TCP_COLLAR_S, normalize, ref_seglst

import scribe  # resolved via the sys.path entry common adds

PUNCT = set(string.punctuation) | {"…", "¿", "¡"}
GAP_S = 0.15


def is_punct(text: str) -> bool:
    return bool(text.strip()) and all(c in PUNCT for c in text.strip())


def words_trim(tokens: list[dict]) -> tuple[scribe.Word, ...]:
    """Group tokens into words; punctuation never starts a word nor extends its end."""
    words: list[scribe.Word] = []
    for tok in tokens:
        text, start, end = tok.get("text", ""), tok.get("start"), tok.get("end")
        if not text or start is None or end is None:
            continue
        if words and (is_punct(text) or not text.startswith(" ")):
            prev = words[-1]
            new_end = prev.end if is_punct(text) else float(end)
            words[-1] = scribe.Word(prev.text + text.strip(), prev.start, new_end)
        elif not is_punct(text):
            words.append(scribe.Word(text.strip(), float(start), float(end)))
    return tuple(w for w in words if w.text)


def transcription(raw: dict, grouping: str) -> list[scribe.TranscriptionSegment]:
    group = scribe._words_from_tokens if grouping == "current" else words_trim  # noqa: SLF001
    segs = []
    for s in raw.get("sentences", []):
        text = s.get("text", "").strip()
        if text:
            segs.append(scribe.TranscriptionSegment(text, float(s["start"]), float(s["end"]), group(s.get("tokens", []))))
    return segs


def drop_tiny(diar: list[scribe.DiarizationSegment], share: float) -> list[scribe.DiarizationSegment]:
    if share <= 0:
        return diar
    total = {}
    for d in diar:
        total[d.speaker] = total.get(d.speaker, 0.0) + d.end - d.start
    whole = sum(total.values()) or 1.0
    keep = {s for s, t in total.items() if t / whole >= share}
    return [d for d in diar if d.speaker in keep]


def word_speakers(diar, transcription_segments) -> list[list[tuple[scribe.Word, str]]]:
    """Per sentence, (word, speaker) pairs using scribe.merge on one-word segments."""
    out = []
    for seg in transcription_segments:
        words = seg.words or (scribe.Word(seg.text, seg.start, seg.end),)
        merged = scribe.merge(diar, [seg])
        # expand merged runs back to words by position
        labels, i = [], 0
        for m in merged:
            n = len(m.text.split(" "))
            labels += [m.speaker] * n
            i += n
        labels = (labels + [labels[-1]] * len(words))[: len(words)] if labels else ["UNKNOWN"] * len(words)
        out.append(list(zip(words, labels, strict=True)))
    return out


def runs(pairs):
    return [(spk, [w for w, _ in grp]) for spk, grp in itertools.groupby(pairs, key=lambda p: p[1])]


def smooth(pairs, mode: str):
    if mode == "none" or len(pairs) < 2:
        return pairs
    k = int(mode.lstrip("edgisland"))
    rs = runs(pairs)
    changed = True
    while changed:
        changed = False
        for i, (spk, ws) in enumerate(rs):
            if len(ws) > k or len(rs) < 2:
                continue
            prev = rs[i - 1] if i > 0 else None
            nxt = rs[i + 1] if i + 1 < len(rs) else None
            target = None
            if prev and nxt and prev[0] == nxt[0]:
                target = prev[0]
            elif mode.startswith("edge"):
                if prev is None and nxt and nxt[1][0].start - ws[-1].end < GAP_S:
                    target = nxt[0]
                elif nxt is None and prev and ws[0].start - prev[1][-1].end < GAP_S:
                    target = prev[0]
            if target and target != spk:
                rs[i] = (target, ws)
                merged = []
                for s, w in rs:
                    if merged and merged[-1][0] == s:
                        merged[-1] = (s, merged[-1][1] + w)
                    else:
                        merged.append((s, w))
                rs = merged
                changed = True
                break
    return [(w, s) for s, ws in rs for w in ws]


def hyp_for(meeting, raw, diar, grouping, tiny, mode):
    segs = transcription(raw, grouping)
    d = drop_tiny(diar, tiny)
    hyp = []
    for pairs in word_speakers(d, segs):
        for spk, ws in runs(smooth(pairs, mode)):
            text = normalize(" ".join(w.text for w in ws))
            if text:
                hyp.append({"session_id": meeting, "speaker": spk, "start_time": ws[0].start, "end_time": ws[-1].end, "words": text})
    return hyp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--asr", default="parakeet-v3-c30")
    parser.add_argument("--diar", default="scribe-e2e")
    parser.add_argument("--variants", default="sdm,ihm")
    parser.add_argument("--meetings", default="all")
    parser.add_argument("--grid", default="full", choices=["full", "edge"])
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    meetings = meetings_for(args.meetings)
    edge_grid = [("trim", 0.0, f"edge{k}") for k in (2, 3, 4, 5, 6)]
    grid = edge_grid if args.grid == "edge" else [
        ("current", 0.0, "none"),
        ("trim", 0.0, "none"),
        ("trim", 0.01, "none"),
        ("trim", 0.02, "none"),
        ("trim", 0.05, "none"),
        ("trim", 0.0, "island1"),
        ("trim", 0.0, "island2"),
        ("trim", 0.0, "edge1"),
        ("trim", 0.0, "edge2"),
        ("trim", 0.02, "island2"),
        ("trim", 0.02, "edge2"),
    ]
    print("| words | tiny | smooth | " + " | ".join(f"{v} cpWER | {v} tcpWER | {v} segs" for v in args.variants.split(",")) + " |")
    print("|" + "---|" * (3 + 3 * len(args.variants.split(","))))
    data = {}
    for v in args.variants.split(","):
        for m in meetings:
            data[v, m] = (
                read_json(asr_run_dir(args.asr, v) / f"{m}.raw.json"),
                [scribe.DiarizationSegment(*s) for s in read_rttm(diar_run_dir(args.diar, v) / f"{m}.rttm")],
            )
    refs = {m: ref_seglst(m) for m in meetings}
    for grouping, tiny, mode in grid:
        cells = []
        for v in args.variants.split(","):
            ref_all, hyp_all = [], []
            for m in meetings:
                raw, diar = data[v, m]
                ref_all += refs[m]
                hyp_all += hyp_for(m, raw, diar, grouping, tiny, mode)
            cp = combine_error_rates(*meeteval.wer.api.cpwer(SegLST(ref_all), SegLST(hyp_all)).values())
            tcp = combine_error_rates(*meeteval.wer.api.tcpwer(SegLST(ref_all), SegLST(hyp_all), collar=TCP_COLLAR_S).values())
            cells += [f"{100 * cp.error_rate:.2f}", f"{100 * tcp.error_rate:.2f}", str(len(hyp_all))]
        print(f"| {grouping} | {tiny:.0%} | {mode} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
