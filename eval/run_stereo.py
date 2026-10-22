"""Per-channel vs mono transcription for synthetic stereo AMI meetings.

Mono pipeline: scribe as it is today (ffmpeg -ac 1, run_transcription + diarize, merge).
Per-channel pipeline: right channel -> run_transcription only, every word "FOCAL"; left
channel -> run_transcription + diarize + merge; the two word streams are interleaved by
time. A gate can drop a channel's words that look like bleed from the other channel.

  uv run python run_stereo.py ami-prep  [--meetings all]   # headsets -> synthetic stereo
  uv run python run_stereo.py ami-run   [--meetings all]   # model runs (cached per channel)
  uv run python run_stereo.py ami-score [--meetings all]

Outputs go to runs/stereo/ami/ (per-channel model outputs, hypotheses) and
runs/results/stereo_*.json.
"""

import argparse
import logging
import subprocess
import sys
import time
import wave
import xml.etree.ElementTree as ET
from pathlib import Path

import meeteval
import numpy as np
from common import DATA_DIR, RESULTS_DIR, RUNS_DIR, meetings_for, read_json, wav_duration, write_json
from meeteval.io import SegLST
from meeteval.wer import combine_error_rates, siso_word_error_rate
from score import TCP_COLLAR_S, normalize, pct, ref_seglst, table

import scribe  # resolved via the sys.path entry common adds

FOCAL = "FOCAL"
HEADSET_DIR = DATA_DIR / "raw" / "headset"
AMI_STEREO_DIR = DATA_DIR / "stereo"
OUT_DIR = RUNS_DIR / "stereo"
SR = 16000
FRAME_S = 0.02
FRAME = int(SR * FRAME_S)
# Word-rank positions of the right-channel focal speaker in each AMI meeting (0 = top talker).
ROTATION_RANKS = (0, 2)


# --- audio helpers ----------------------------------------------------------------


def read_wav(path: Path) -> np.ndarray:
    """Return float32 samples shaped (n,) for mono or (n, channels)."""
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SR, path
        ch = wf.getnchannels()
        x = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return x.reshape(-1, ch) if ch > 1 else x


def write_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ch = 1 if x.ndim == 1 else x.shape[1]
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


def ffmpeg(src: Path, dst: Path, *args: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-vn", "-acodec", "pcm_s16le",
         "-ar", str(SR), *args, str(tmp)],
        check=True,
    )  # fmt: skip
    tmp.rename(dst)


def split_channels(stereo: Path, left: Path, right: Path) -> None:
    """Split a stereo wav into mono channel wavs the way scribe would (ffmpeg pan)."""
    if not left.exists():
        ffmpeg(stereo, left, "-af", "pan=mono|c0=c0")
    if not right.exists():
        ffmpeg(stereo, right, "-af", "pan=mono|c0=c1")


def frame_db(x: np.ndarray) -> np.ndarray:
    """Per-frame (20 ms) power in dBFS."""
    n = len(x) // FRAME
    p = np.square(x[: n * FRAME].reshape(n, FRAME), dtype=np.float64).mean(axis=1)
    return 10 * np.log10(p + 1e-12)


def span_db(db: np.ndarray, start: float, end: float) -> float:
    """Mean power (dB) of the frames covering [start, end]."""
    a = max(int(start / FRAME_S), 0)
    b = min(max(int(np.ceil(end / FRAME_S)), a + 1), len(db))
    if a >= b:
        return -120.0
    return float(10 * np.log10(np.mean(10 ** (db[a:b] / 10)) + 1e-12))


# --- model runs (cached) ----------------------------------------------------------


def asr_words(segments: list[scribe.TranscriptionSegment]) -> list[dict]:
    return [
        {"text": s.text, "start": s.start, "end": s.end,
         "words": [{"text": w.text, "start": w.start, "end": w.end} for w in s.words]}
        for s in segments
    ]  # fmt: skip


def run_asr(wav: Path, out: Path) -> dict:
    if out.exists():
        return read_json(out)
    t0 = time.perf_counter()
    segments = scribe.run_transcription(wav)
    data = {"segments": asr_words(segments), "runtime_s": round(time.perf_counter() - t0, 2),
            "audio_s": round(wav_duration(wav), 2)}  # fmt: skip
    write_json(out, data)
    print(f"  asr {out.stem}: {data['runtime_s']} s for {data['audio_s']} s", file=sys.stderr, flush=True)
    return data


def run_diar(wav: Path, out: Path) -> dict:
    if out.exists():
        return read_json(out)
    t0 = time.perf_counter()
    segments = scribe.diarize(wav)
    data = {"segments": [list(s) for s in segments], "runtime_s": round(time.perf_counter() - t0, 2),
            "audio_s": round(wav_duration(wav), 2)}  # fmt: skip
    write_json(out, data)
    print(f"  diar {out.stem}: {data['runtime_s']} s, {len({s[0] for s in data['segments']})} speakers",
          file=sys.stderr, flush=True)  # fmt: skip
    return data


# --- hypothesis building + gating --------------------------------------------------


def _segment_words(seg: dict) -> list[dict]:
    return seg["words"] or [{"text": seg["text"], "start": seg["start"], "end": seg["end"]}]


def word_levels(asr: dict, db_own: np.ndarray, db_other: np.ndarray) -> list[list[tuple[float, float]]]:
    """Per word: (own-channel dB, other-channel dB) over the word's span."""
    return [
        [(span_db(db_own, w["start"], w["end"]), span_db(db_other, w["start"], w["end"])) for w in _segment_words(s)]
        for s in asr["segments"]
    ]


def speech_level(levels: list[list[tuple[float, float]]]) -> float:
    """A channel's speech level: the median own-channel dB of its words."""
    own = [o for seg in levels for o, _ in seg]
    return float(np.median(own)) if own else -120.0


def keep_masks(
    levels: list[list[tuple[float, float]]],
    level_own: float,
    level_other: float,
    gate: dict | None,
    thr_own: float = -120.0,
) -> list[list[bool]]:
    """Gate words: drop if weak for their channel, or if the other channel dominates.

    gate = {"weak": dB below the channel's speech level, "dom": dB by which the other
    channel's level (relative to its own speech level) exceeds this channel's,
    "act": True to drop words whose own-channel level is under the channel's
    speech-activity threshold ``thr_own``}.
    """
    if not gate:
        return [[True] * len(seg) for seg in levels]
    masks = []
    for seg in levels:
        mask = []
        for own, other in seg:
            rel_own, rel_other = own - level_own, other - level_other
            ok = True
            if gate.get("weak") is not None and rel_own < -gate["weak"]:
                ok = False
            if gate.get("dom") is not None and rel_other - rel_own > gate["dom"]:
                ok = False
            if gate.get("act") and own < thr_own:
                ok = False
            mask.append(ok)
        masks.append(mask)
    return masks


def _kept_runs(seg: dict, mask: list[bool]) -> list[list[scribe.Word]]:
    runs: list[list[scribe.Word]] = [[]]
    for w, keep in zip(_segment_words(seg), mask, strict=True):
        if keep:
            runs[-1].append(scribe.Word(w["text"], w["start"], w["end"]))
        elif runs[-1]:
            runs.append([])
    return [r for r in runs if r]


def mono_merged(asr: dict, diar: dict) -> list[scribe.MergedSegment]:
    diarization = [scribe.DiarizationSegment(*s) for s in diar["segments"]]
    transcription = [
        scribe.TranscriptionSegment(s["text"], s["start"], s["end"], tuple(scribe.Word(**w) for w in s["words"]))
        for s in asr["segments"]
    ]
    return scribe.merge(diarization, transcription)


def per_channel_merged(
    right: dict, left: dict, left_diar: dict, right_mask: list[list[bool]], left_mask: list[list[bool]]
) -> list[scribe.MergedSegment]:
    """Right words all FOCAL; left words through scribe.merge; interleaved by start time."""
    merged = []
    for seg, mask in zip(right["segments"], right_mask, strict=True):
        for run in _kept_runs(seg, mask):
            merged.append(scribe.MergedSegment(FOCAL, " ".join(w.text for w in run), run[0].start, run[-1].end))
    merged += left_merge(left, left_diar, left_mask)
    return sorted(merged, key=lambda m: (m.start, m.end))


def left_merge(left: dict, left_diar: dict, left_mask: list[list[bool]]) -> list[scribe.MergedSegment]:
    """The left channel's kept words through scribe.merge with the left-channel diarization."""
    left_ts = []
    for seg, mask in zip(left["segments"], left_mask, strict=True):
        for run in _kept_runs(seg, mask):
            left_ts.append(scribe.TranscriptionSegment(" ".join(w.text for w in run), run[0].start, run[-1].end, tuple(run)))
    diarization = [scribe.DiarizationSegment(*s) for s in left_diar["segments"]]
    return scribe.merge(diarization, left_ts)


def to_seglst(merged: list[scribe.MergedSegment], session: str) -> list[dict]:
    hyp = [
        {"session_id": session, "speaker": m.speaker, "start_time": m.start, "end_time": m.end, "words": normalize(m.text)}
        for m in merged
    ]
    return [h for h in hyp if h["words"]]


# --- AMI --------------------------------------------------------------------------


def ami_channel_speakers(meeting: str) -> dict[int, str]:
    root = ET.parse(DATA_DIR / "raw" / "manual" / "corpusResources" / "meetings.xml").getroot()
    for m in root.iter("meeting"):
        if m.get("observation") == meeting:
            return {int(s.get("channel")): s.get("global_name") for s in m.iter("speaker")}
    raise KeyError(meeting)


def ami_rotations(meeting: str) -> list[str]:
    counts: dict[str, int] = {}
    for seg in ref_seglst(meeting):
        counts[seg["speaker"]] = counts.get(seg["speaker"], 0) + len(seg["words"].split())
    ranked = sorted(counts, key=lambda s: -counts[s])
    return [ranked[r] for r in ROTATION_RANKS]


def ami_paths(meeting: str, spk: str) -> dict[str, Path]:
    base = AMI_STEREO_DIR / meeting
    return {"stereo": base / f"{spk}.stereo.wav", "L": base / f"{spk}.L.wav", "R": base / f"{spk}.R.wav",
            "mono": base / "mono.wav"}  # fmt: skip


def ami_prep(meetings: list[str]) -> None:
    check = {}
    for meeting in meetings:
        chan = ami_channel_speakers(meeting)
        heads = {chan[k]: read_wav(HEADSET_DIR / f"{meeting}.Headset-{k}.wav") for k in sorted(chan)}
        n = max(len(x) for x in heads.values())
        heads = {s: np.pad(x, (0, n - len(x))) for s, x in heads.items()}
        total = sum(heads.values())
        # One gain per meeting so every rotation's L + R is the same full mix.
        peak = max([np.abs(total).max()] + [np.abs(x).max() for x in heads.values()]
                   + [np.abs(total - x).max() for x in heads.values()])  # fmt: skip
        gain = float(0.9 / peak)
        # Check the channel -> speaker map: each headset should be loudest during its
        # own speaker's reference segments (headsets pick up neighbours' speech).
        dbs = {s: frame_db(x * gain) for s, x in heads.items()}
        loud = {}
        for spk in heads:
            segs = [(g["start_time"], g["end_time"]) for g in ref_seglst(meeting) if g["speaker"] == spk]
            loud[spk] = {h: round(float(np.median([span_db(dbs[h], a, b) for a, b in segs])), 1) for h in heads}
        check[meeting] = {"gain": gain, "median_db_by_ref_speaker": loud,
                          "map_ok": all(max(loud, key=lambda s, h=h: loud[s][h]) == h for h in heads)}  # fmt: skip
        print(meeting, "map ok" if check[meeting]["map_ok"] else f"MAP MISMATCH {loud}", file=sys.stderr)
        for spk in ami_rotations(meeting):
            p = ami_paths(meeting, spk)
            if p["stereo"].exists():
                continue
            right = heads[spk] * gain
            left = (total - heads[spk]) * gain
            write_wav(p["stereo"], np.stack([left, right], axis=1))
            split_channels(p["stereo"], p["L"], p["R"])
            if not p["mono"].exists():
                ffmpeg(p["stereo"], p["mono"], "-ac", "1")  # scribe's downmix
    write_json(OUT_DIR / "ami" / "prep_check.json", check)


def ami_run(meetings: list[str]) -> None:
    out = OUT_DIR / "ami" / "raw"
    for meeting in meetings:
        rotations = ami_rotations(meeting)
        mono = ami_paths(meeting, rotations[0])["mono"]
        run_asr(mono, out / f"{meeting}.mono.asr.json")
        run_diar(mono, out / f"{meeting}.mono.diar.json")
        for spk in rotations:
            p = ami_paths(meeting, spk)
            run_asr(p["R"], out / f"{meeting}.{spk}.R.asr.json")
            run_asr(p["L"], out / f"{meeting}.{spk}.L.asr.json")
            run_diar(p["L"], out / f"{meeting}.{spk}.L.diar.json")


GATES = {
    "none": None,
    "dom6": {"dom": 6},
    "dom10": {"dom": 10},
    "dom15": {"dom": 15},
    "weak20": {"weak": 20},
    "act": {"act": True},
    "act+dom10": {"act": True, "dom": 10},
}


def channel_levels(asr: dict, db_own: np.ndarray, db_other: np.ndarray) -> tuple[list, float]:
    levels = word_levels(asr, db_own, db_other)
    return levels, speech_level(levels)


def _speaker_wer(ref: list[dict], hyp: list[dict], ref_spk: str, hyp_spk: str | None):  # noqa: ANN202
    r = " ".join(s["words"] for s in sorted(ref, key=lambda s: s["start_time"]) if s["speaker"] == ref_spk)
    h = " ".join(s["words"] for s in sorted(hyp, key=lambda s: s["start_time"]) if s["speaker"] == hyp_spk)
    return siso_word_error_rate(r, h)


def _others_cp(ref: list[dict], hyp: list[dict], ref_spk: str, hyp_spk: str | None):  # noqa: ANN202
    r = SegLST([s for s in ref if s["speaker"] != ref_spk])
    h = SegLST([s for s in hyp if s["speaker"] != hyp_spk])
    return next(iter(meeteval.wer.api.cpwer(r, h).values()))


def _score_sessions(pairs: dict[str, tuple[list[dict], list[dict], str, str | None]]) -> dict:
    """pairs: session -> (ref, hyp, focal ref speaker, focal hyp label or None to take cpWER's)."""
    ref_all = SegLST([s for r, _, _, _ in pairs.values() for s in r])
    hyp_all = SegLST([s for _, h, _, _ in pairs.values() for s in h])
    cp = meeteval.wer.api.cpwer(ref_all, hyp_all)
    tcp = meeteval.wer.api.tcpwer(ref_all, hyp_all, collar=TCP_COLLAR_S)
    per, focal_ers, oth_ers = {}, [], []
    for session, (ref, hyp, ref_spk, hyp_spk) in pairs.items():
        if hyp_spk is None:
            hyp_spk = dict(cp[session].assignment).get(ref_spk)
        focal = _speaker_wer(ref, hyp, ref_spk, hyp_spk)
        oth = _others_cp(ref, hyp, ref_spk, hyp_spk)
        focal_ers.append(focal)
        oth_ers.append(oth)
        per[session] = {"cpwer": cp[session].error_rate, "tcpwer": tcp[session].error_rate, "focal_wer": focal.error_rate,
                        "others_cpwer": oth.error_rate, "hyp_speakers": len({s["speaker"] for s in hyp}),
                        "cp_del": cp[session].deletions / cp[session].length,
                        "cp_ins": cp[session].insertions / cp[session].length}  # fmt: skip
    cpc, tcpc = combine_error_rates(*cp.values()), combine_error_rates(*tcp.values())
    focal_c, othc = combine_error_rates(*focal_ers), combine_error_rates(*oth_ers)
    pooled = {"cpwer": cpc.error_rate, "tcpwer": tcpc.error_rate, "focal_wer": focal_c.error_rate,
              "others_cpwer": othc.error_rate, "cp_sub": cpc.substitutions / cpc.length,
              "cp_del": cpc.deletions / cpc.length, "cp_ins": cpc.insertions / cpc.length}  # fmt: skip
    return {"per_session": per, "pooled": pooled}


def ami_score(meetings: list[str]) -> None:
    raw = OUT_DIR / "ami" / "raw"
    hyp_dir = OUT_DIR / "ami" / "hyp"
    systems: dict[str, dict] = {"mono": {}} | {f"per-channel/{g}": {} for g in GATES}
    runtime = {"mono": [0.0, 0.0, 0.0], "per-channel": [0.0, 0.0, 0.0, 0.0]}  # [asr, diar, audio], [R asr, L asr, L diar, audio]
    bleeds: dict[str, dict] = {}
    for meeting in meetings:
        refs = ref_seglst(meeting)
        mono_asr, mono_diar = read_json(raw / f"{meeting}.mono.asr.json"), read_json(raw / f"{meeting}.mono.diar.json")
        mono_hyp = to_seglst(mono_merged(mono_asr, mono_diar), meeting)
        write_json(hyp_dir / "mono" / f"{meeting}.json", mono_hyp)
        for spk in ami_rotations(meeting):
            session = f"{meeting}~{spk}"
            ref = [{**s, "session_id": session} for s in refs]
            systems["mono"][session] = (ref, [{**h, "session_id": session} for h in mono_hyp], spk, None)
            runtime["mono"][0] += mono_asr["runtime_s"]
            runtime["mono"][1] += mono_diar["runtime_s"]
            runtime["mono"][2] += mono_asr["audio_s"]
            right = read_json(raw / f"{meeting}.{spk}.R.asr.json")
            left = read_json(raw / f"{meeting}.{spk}.L.asr.json")
            left_diar = read_json(raw / f"{meeting}.{spk}.L.diar.json")
            runtime["per-channel"][0] += right["runtime_s"]
            runtime["per-channel"][1] += left["runtime_s"]
            runtime["per-channel"][2] += left_diar["runtime_s"]
            runtime["per-channel"][3] += right["audio_s"]
            p = ami_paths(meeting, spk)
            fr = channel_frames(p["stereo"])
            bleeds[session] = bleed_stats(fr)
            db_l, db_r, thr_l, thr_r = fr["db_l"], fr["db_r"], fr["thr_l"], fr["thr_r"]
            r_lv, r_level = channel_levels(right, db_r, db_l)
            l_lv, l_level = channel_levels(left, db_l, db_r)
            for g, gate in GATES.items():
                merged = per_channel_merged(right, left, left_diar, keep_masks(r_lv, r_level, l_level, gate, thr_r),
                                            keep_masks(l_lv, l_level, r_level, gate, thr_l))  # fmt: skip
                hyp = to_seglst(merged, session)
                write_json(hyp_dir / f"per-channel-{g}" / f"{session}.json", hyp)
                systems[f"per-channel/{g}"][session] = (ref, hyp, spk, FOCAL)
    results = {name: _score_sessions(pairs) for name, pairs in systems.items()}
    results["runtime"] = runtime
    results["bleed"] = bleeds
    write_json(RESULTS_DIR / f"stereo_ami__{'+'.join(meetings) if len(meetings) < 8 else 'all'}.json", results)
    header = ["system", "cpWER %", "tcpWER %", "focal WER %", "others cpWER %", "sub %", "del %", "ins %"]
    rows = [[n, pct(r["pooled"]["cpwer"]), pct(r["pooled"]["tcpwer"]), pct(r["pooled"]["focal_wer"]),
             pct(r["pooled"]["others_cpwer"]), pct(r["pooled"]["cp_sub"]), pct(r["pooled"]["cp_del"]),
             pct(r["pooled"]["cp_ins"])] for n, r in results.items() if n not in ("runtime", "bleed")]  # fmt: skip
    print(table(header, rows))
    header = ["session", "mono cp", "per-ch none cp", "per-ch dom10 cp", "mono tcp", "per-ch dom10 tcp",
              "mono focal", "per-ch dom10 focal", "mono spk", "per-ch spk"]  # fmt: skip
    rows = []
    for s in results["mono"]["per_session"]:
        m, n, d = (results[k]["per_session"][s] for k in ("mono", "per-channel/none", "per-channel/dom10"))
        rows.append([s, pct(m["cpwer"]), pct(n["cpwer"]), pct(d["cpwer"]), pct(m["tcpwer"]), pct(d["tcpwer"]),
                     pct(m["focal_wer"]), pct(d["focal_wer"]), m["hyp_speakers"], d["hyp_speakers"]])  # fmt: skip
    print(table(header, rows))
    print(runtime)


# --- channel activity and leakage --------------------------------------------------


def active_threshold(db: np.ndarray) -> float:
    """Speech-activity threshold for one channel: 6 dB over its floor, at most 40 dB under its peak."""
    live = db[db > -110]  # Ignore digital silence on an idle channel.
    if not len(live):
        return 0.0
    return float(max(np.percentile(live, 10) + 6, np.percentile(live, 99) - 40))


def smooth_db(db: np.ndarray, frames: int = 15) -> np.ndarray:
    p = np.convolve(10 ** (db / 10), np.ones(frames) / frames, mode="same")
    return 10 * np.log10(p + 1e-12)


def channel_frames(stereo: Path, vad: tuple[Path, Path] | None = None) -> dict:
    """Per-frame channel levels and speech activity.

    Activity comes from Silero VAD segments per channel when ``vad`` (left, right JSON)
    is given, otherwise from an energy threshold on 300 ms power averages (so syllable
    gaps, often digital silence on the noise-suppressed remote channel, do not split
    speech). The energy thresholds are kept either way for the gates.
    """
    x = read_wav(stereo)
    db_l, db_r = frame_db(x[:, 0]), frame_db(x[:, 1])
    thr_l, thr_r = active_threshold(db_l), active_threshold(db_r)
    if vad:
        act_l, act_r = (vad_mask(read_json(v), len(db_l)) for v in vad)
    else:
        act_l, act_r = smooth_db(db_l) > thr_l, smooth_db(db_r) > thr_r
    return {"db_l": db_l, "db_r": db_r, "thr_l": thr_l, "thr_r": thr_r, "act_l": act_l, "act_r": act_r,
            "identical": bool(np.array_equal(x[:, 0], x[:, 1]))}  # fmt: skip


def vad_mask(segments: list[list[float]], n: int) -> np.ndarray:
    mask = np.zeros(n, bool)
    for start, end in segments:
        mask[int(start / FRAME_S) : int(np.ceil(end / FRAME_S))] = True
    return mask


def _q(a: np.ndarray) -> list[float]:
    return [round(float(v), 1) for v in np.percentile(a, [5, 25, 50, 75, 95])] if len(a) else []


def bleed_stats(fr: dict) -> dict:
    """Levels of each channel while only the other one has speech.

    If the right-channel speaker leaks into the left channel, the left level during
    right-only frames would sit above its idle level and follow the right level.
    """
    db_l, db_r, act_l, act_r = smooth_db(fr["db_l"]), smooth_db(fr["db_r"]), fr["act_l"], fr["act_r"]
    r_only, l_only, idle, both = act_r & ~act_l, act_l & ~act_r, ~act_l & ~act_r, act_l & act_r
    out = {"frames_s": {k: round(float(v.sum()) * FRAME_S, 1) for k, v in
                        {"r_only": r_only, "l_only": l_only, "both": both, "idle": idle}.items()}}  # fmt: skip
    out["thr_db"] = {"L": round(fr["thr_l"], 1), "R": round(fr["thr_r"], 1)}
    out["r_minus_l_r_only"] = _q(db_r[r_only] - db_l[r_only])
    out["r_minus_l_l_only"] = _q(db_r[l_only] - db_l[l_only])
    # Leak = channel level while only the other side talks, minus its level when idle.
    out["left_leak_db"] = round(float(np.median(db_l[r_only]) - np.median(db_l[idle])), 1) if r_only.any() else None
    out["right_leak_db"] = round(float(np.median(db_r[l_only]) - np.median(db_r[idle])), 1) if l_only.any() else None
    # Echo coupling: does the quiet channel's level track the talking channel's level?
    out["corr_l_r_on_r_only"] = round(float(np.corrcoef(db_r[r_only], db_l[r_only])[0, 1]), 3) if r_only.sum() > 10 else None
    out["corr_l_r_on_l_only"] = round(float(np.corrcoef(db_l[l_only], db_r[l_only])[0, 1]), 3) if l_only.sum() > 10 else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("ami-prep", "ami-run", "ami-score"):
        sub.add_parser(name).add_argument("--meetings", default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)  # meeteval ordering/self-overlap warnings
    if args.cmd == "ami-prep":
        ami_prep(meetings_for(args.meetings))
    elif args.cmd == "ami-run":
        ami_run(meetings_for(args.meetings))
    elif args.cmd == "ami-score":
        ami_score(meetings_for(args.meetings))


if __name__ == "__main__":
    main()
