"""Score AMI eval candidates.

  uv run python score.py asr   --name parakeet-v3 [--variant sdm|ihm|both] [--meetings all|quick|M1,M2]
  uv run python score.py diar  --name senko       [--variant ...] [--meetings ...]
  uv run python score.py combo --asr parakeet-v3 --diar senko --merge word|sentence|both [--variant ...]

Prints markdown tables and writes JSON to runs/results/<kind>__<names>__<variant>__<subset>.json.
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)  # whisper_normalizer docstring escapes

import meeteval
from common import (
    DATA_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    asr_run_dir,
    diar_run_dir,
    meetings_for,
    read_json,
    read_rttm,
    ref_rttm_path,
    ref_seglst_path,
    ref_uem_path,
    write_json,
)
from meeteval.io import SegLST
from meeteval.wer import combine_error_rates, siso_word_error_rate
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from whisper_normalizer.english import EnglishTextNormalizer

import scribe  # resolved via the sys.path entry common adds

TCP_COLLAR_S = 5.0
_normalize = EnglishTextNormalizer()


def normalize(text: str) -> str:
    return " ".join(_normalize(text).split())


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def variants(arg: str) -> list[str]:
    return ["sdm", "ihm"] if arg == "both" else [arg]


def results_path(kind: str, names: str, variant: str, subset: str) -> Path:
    return RESULTS_DIR / f"{kind}__{names}__{variant}__{subset.replace(',', '+')}.json"


def ref_seglst(meeting: str) -> list[dict]:
    """Reference segments with normalized words (empty ones dropped)."""
    segments = []
    for seg in read_json(ref_seglst_path(meeting)):
        words = normalize(seg["words"])
        if words:
            segments.append({**seg, "words": words})
    return segments


def _run_meta(metas: list[dict]) -> dict:
    audio = sum(m["audio_s"] for m in metas)
    runtime = sum(m["runtime_s"] for m in metas)
    return {
        "runtime_s": round(runtime, 1),
        "audio_s": round(audio, 1),
        "rtfx": round(audio / runtime, 1) if runtime else None,
        "peak_mem_mb_max": max(m["peak_mem_mb"] for m in metas),
    }


# --- ASR -------------------------------------------------------------------------


def hyp_text(asr: dict) -> str:
    if asr.get("sentences"):
        return " ".join(normalize(s["text"]) for s in asr["sentences"])
    return normalize(" ".join(w["text"] for w in asr["words"]))


def score_asr(name: str, variant: str, meetings: list[str]) -> dict:
    per_meeting = {}
    errors = []
    metas = []
    for meeting in meetings:
        asr = read_json(asr_run_dir(name, variant) / f"{meeting}.json")
        ref = " ".join(s["words"] for s in ref_seglst(meeting))  # utterances in start-time order
        er = siso_word_error_rate(ref, hyp_text(asr))
        errors.append(er)
        metas.append(asr["meta"])
        per_meeting[meeting] = _er_dict(er) | {"runtime_s": asr["meta"]["runtime_s"], "peak_mem_mb": asr["meta"]["peak_mem_mb"]}
    return {"per_meeting": per_meeting, "pooled": _er_dict(combine_error_rates(*errors)), "run": _run_meta(metas)}


def _er_dict(er) -> dict:  # noqa: ANN001
    n = er.length or 1
    return {
        "wer": er.error_rate,
        "sub": er.substitutions / n,
        "del": er.deletions / n,
        "ins": er.insertions / n,
        "errors": er.errors,
        "ref_words": er.length,
    }


def print_asr(title: str, result: dict) -> None:
    header = ["meeting", "ref words", "WER %", "sub %", "del %", "ins %", "runtime s", "peak MB"]
    rows = [
        [m, r["ref_words"], pct(r["wer"]), pct(r["sub"]), pct(r["del"]), pct(r["ins"]), r["runtime_s"], r["peak_mem_mb"]]
        for m, r in result["per_meeting"].items()
    ]
    p, run = result["pooled"], result["run"]
    rows.append(["**pooled**", p["ref_words"], f"**{pct(p['wer'])}**", pct(p["sub"]), pct(p["del"]), pct(p["ins"]),
                 f"{run['runtime_s']} (RTFx {run['rtfx']})", f"max {run['peak_mem_mb_max']}"])  # fmt: skip
    print(f"\n### {title}\n\n{table(header, rows)}")


# --- Diarization -----------------------------------------------------------------


def annotation(segments: list[tuple[str, float, float]], uri: str) -> Annotation:
    ann = Annotation(uri=uri)
    for i, (speaker, start, end) in enumerate(segments):
        ann[Segment(start, end), i] = speaker
    return ann


def uem_timeline(meeting: str) -> Timeline:
    fields = ref_uem_path(meeting).read_text().split()
    return Timeline([Segment(float(fields[2]), float(fields[3]))], uri=meeting)


def score_diar(name: str, variant: str, meetings: list[str]) -> dict:
    # pyannote's collar is the total width around each reference boundary, so
    # collar=0.5 is the conventional "collar 0.25" (+/-0.25 s, md-eval -c 0.25).
    der0 = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    der25 = DiarizationErrorRate(collar=0.5, skip_overlap=False)
    jer = JaccardErrorRate(collar=0.0, skip_overlap=False)
    per_meeting = {}
    metas = []
    for meeting in meetings:
        run_dir = diar_run_dir(name, variant)
        ref_segments = read_rttm(ref_rttm_path(meeting))
        hyp_segments = read_rttm(run_dir / f"{meeting}.rttm")
        ref, hyp = annotation(ref_segments, meeting), annotation(hyp_segments, meeting)
        uem = uem_timeline(meeting)
        d0 = der0(ref, hyp, uem=uem, detailed=True)
        d25 = der25(ref, hyp, uem=uem, detailed=True)
        j = jer(ref, hyp, uem=uem)
        n_ref, n_hyp = len({s for s, _, _ in ref_segments}), len({s for s, _, _ in hyp_segments})
        meta = read_json(run_dir / f"{meeting}.meta.json")
        metas.append(meta)
        per_meeting[meeting] = {
            "der_c0": d0["diarization error rate"],
            "der_c25": d25["diarization error rate"],
            "miss_c0": d0["missed detection"] / d0["total"],
            "fa_c0": d0["false alarm"] / d0["total"],
            "conf_c0": d0["confusion"] / d0["total"],
            "jer": j,
            "ref_speakers": n_ref,
            "hyp_speakers": n_hyp,
            "spk_count_err": n_hyp - n_ref,
            "runtime_s": meta["runtime_s"],
            "peak_mem_mb": meta["peak_mem_mb"],
        }
    pooled = {
        "der_c0": abs(der0),
        "der_c25": abs(der25),
        "miss_c0": der0["missed detection"] / der0["total"],
        "fa_c0": der0["false alarm"] / der0["total"],
        "conf_c0": der0["confusion"] / der0["total"],
        "jer": abs(jer),
        "spk_count_mae": sum(abs(r["spk_count_err"]) for r in per_meeting.values()) / len(per_meeting),
        "spk_count_exact": sum(r["spk_count_err"] == 0 for r in per_meeting.values()),
    }
    return {"per_meeting": per_meeting, "pooled": pooled, "run": _run_meta(metas)}


def print_diar(title: str, result: dict) -> None:
    header = ["meeting", "DER c0 %", "DER c.25 %", "miss %", "FA %", "conf %", "JER %", "spk ref/hyp", "runtime s", "peak MB"]
    rows = [
        [m, pct(r["der_c0"]), pct(r["der_c25"]), pct(r["miss_c0"]), pct(r["fa_c0"]), pct(r["conf_c0"]), pct(r["jer"]),
         f"{r['ref_speakers']}/{r['hyp_speakers']}", r["runtime_s"], r["peak_mem_mb"]]
        for m, r in result["per_meeting"].items()
    ]  # fmt: skip
    p, run = result["pooled"], result["run"]
    rows.append(["**pooled**", f"**{pct(p['der_c0'])}**", f"**{pct(p['der_c25'])}**", pct(p["miss_c0"]), pct(p["fa_c0"]),
                 pct(p["conf_c0"]), pct(p["jer"]), f"MAE {p['spk_count_mae']:.2f}, exact {p['spk_count_exact']}/{len(result['per_meeting'])}",
                 f"{run['runtime_s']} (RTFx {run['rtfx']})", f"max {run['peak_mem_mb_max']}"])  # fmt: skip
    print(f"\n### {title}\n\n{table(header, rows)}")


# --- Combo (ASR + diarization via scribe.merge) --------------------------------------


def transcription_from_asr(asr: dict, merge_mode: str) -> list[scribe.TranscriptionSegment]:
    """Build scribe TranscriptionSegments; merge_mode 'sentence' drops word timings (old merge)."""
    sentences = asr.get("sentences") or []
    if not sentences:
        if merge_mode == "sentence":
            msg = "merge=sentence needs 'sentences' in the ASR JSON"
            raise SystemExit(msg)
        words = asr["words"]
        sentences = [{"text": " ".join(w["text"] for w in words), "start": words[0]["start"], "end": words[-1]["end"], "words": words}]
    segments = []
    for s in sentences:
        words = () if merge_mode == "sentence" else tuple(scribe.Word(w["text"], w["start"], w["end"]) for w in s.get("words", []))
        segments.append(scribe.TranscriptionSegment(s["text"], s["start"], s["end"], words))
    return segments


def combo_hyp(asr_name: str, diar_name: str, variant: str, merge_mode: str, meeting: str) -> list[dict]:
    asr = read_json(asr_run_dir(asr_name, variant) / f"{meeting}.json")
    diarization = [scribe.DiarizationSegment(*s) for s in read_rttm(diar_run_dir(diar_name, variant) / f"{meeting}.rttm")]
    merged = scribe.merge(diarization, transcription_from_asr(asr, merge_mode))
    hyp = [
        {"session_id": meeting, "speaker": m.speaker, "start_time": m.start, "end_time": m.end, "words": normalize(m.text)}
        for m in merged
    ]
    hyp = [h for h in hyp if h["words"]]
    write_json(RUNS_DIR / "combo" / f"{asr_name}+{diar_name}" / variant / merge_mode / f"{meeting}.json", hyp)
    return hyp


def score_combo(asr_name: str, diar_name: str, variant: str, merge_mode: str, meetings: list[str]) -> dict:
    ref_all, hyp_all = [], []
    for meeting in meetings:
        ref_all += ref_seglst(meeting)
        hyp_all += combo_hyp(asr_name, diar_name, variant, merge_mode, meeting)
    ref, hyp = SegLST(ref_all), SegLST(hyp_all)
    cp = meeteval.wer.api.cpwer(ref, hyp)
    tcp = meeteval.wer.api.tcpwer(ref, hyp, collar=TCP_COLLAR_S)
    orc = meeteval.wer.api.greedy_orcwer(ref, hyp)
    per_meeting = {
        m: {"cpwer": cp[m].error_rate, "tcpwer": tcp[m].error_rate, "orcwer_greedy": orc[m].error_rate,
            "cp_sub": cp[m].substitutions / cp[m].length, "cp_del": cp[m].deletions / cp[m].length,
            "cp_ins": cp[m].insertions / cp[m].length, "ref_words": cp[m].length,
            "missed_speaker": cp[m].missed_speaker, "falarm_speaker": cp[m].falarm_speaker}
        for m in meetings
    }  # fmt: skip
    cpc, tcpc, orcc = (combine_error_rates(*r.values()) for r in (cp, tcp, orc))
    pooled = {
        "cpwer": cpc.error_rate, "tcpwer": tcpc.error_rate, "orcwer_greedy": orcc.error_rate,
        "cp_sub": cpc.substitutions / cpc.length, "cp_del": cpc.deletions / cpc.length,
        "cp_ins": cpc.insertions / cpc.length, "ref_words": cpc.length,
    }  # fmt: skip
    return {"per_meeting": per_meeting, "pooled": pooled}


def print_combo(title: str, result: dict) -> None:
    header = ["meeting", "cpWER %", "tcpWER %", "ORC-WER % (greedy)", "sub %", "del %", "ins %", "missed/FA spk"]
    rows = [
        [m, pct(r["cpwer"]), pct(r["tcpwer"]), pct(r["orcwer_greedy"]), pct(r["cp_sub"]), pct(r["cp_del"]), pct(r["cp_ins"]),
         f"{r['missed_speaker']}/{r['falarm_speaker']}"]
        for m, r in result["per_meeting"].items()
    ]  # fmt: skip
    p = result["pooled"]
    rows.append(["**pooled**", f"**{pct(p['cpwer'])}**", f"**{pct(p['tcpwer'])}**", pct(p["orcwer_greedy"]),
                 pct(p["cp_sub"]), pct(p["cp_del"]), pct(p["cp_ins"]), ""])  # fmt: skip
    print(f"\n### {title}\n\n{table(header, rows)}")


# --- CLI -------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Score AMI eval candidates")
    sub = parser.add_subparsers(dest="kind", required=True)
    for kind in ("asr", "diar", "combo"):
        p = sub.add_parser(kind)
        p.add_argument("--variant", default="both", choices=["sdm", "ihm", "both"])
        p.add_argument("--meetings", default="all", help="all, quick, or comma-separated meeting ids")
        if kind == "combo":
            p.add_argument("--asr", required=True)
            p.add_argument("--diar", required=True)
            p.add_argument("--merge", default="both", choices=["word", "sentence", "both"])
        else:
            p.add_argument("--name", required=True)
    parser.add_argument("-v", "--verbose", action="store_true", help="show meeteval warnings")
    args = parser.parse_args()
    # meeteval warns about hypothesis self-overlap and pseudo-word-timing order; these are
    # expected for merged segments and noisy, so they are hidden unless -v.
    logging.basicConfig(level=logging.WARNING if args.verbose else logging.ERROR)
    meetings = meetings_for(args.meetings)
    if not ref_seglst_path(meetings[0]).exists():
        sys.exit(f"missing references under {DATA_DIR}; run prepare_ami.py first")

    for variant in variants(args.variant):
        if args.kind == "asr":
            result = score_asr(args.name, variant, meetings)
            print_asr(f"ASR {args.name} / {variant} / {args.meetings}", result)
            write_json(results_path("asr", args.name, variant, args.meetings), result)
        elif args.kind == "diar":
            result = score_diar(args.name, variant, meetings)
            print_diar(f"Diarization {args.name} / {variant} / {args.meetings}", result)
            write_json(results_path("diar", args.name, variant, args.meetings), result)
        else:
            for merge_mode in ["word", "sentence"] if args.merge == "both" else [args.merge]:
                result = score_combo(args.asr, args.diar, variant, merge_mode, meetings)
                title = f"Combo {args.asr} + {args.diar} / merge={merge_mode} / {variant} / {args.meetings}"
                print_combo(title, result)
                names = f"{args.asr}+{args.diar}+{merge_mode}"
                write_json(results_path("combo", names, variant, args.meetings), result)


if __name__ == "__main__":
    main()
