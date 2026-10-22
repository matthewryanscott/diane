"""Local transcription and speaker diarization with parakeet and Nemotron 3 (or senko)."""

import argparse
import json
import os
import shutil
import string
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np

UNKNOWN_SPEAKER = "UNKNOWN"
# A word that overlaps no diarization segment takes the nearest speaker within this gap.
_NEAREST_SPEAKER_TOLERANCE_S = 1.0
# Speaker runs this short are smoothed unless a pause this long backs the switch.
_SMOOTH_MAX_WORDS = 3
_SMOOTH_MIN_PAUSE_S = 0.15
_PUNCTUATION = frozenset(string.punctuation) | {"…", "¿", "¡"}
# A subsegment embeds as one speaker's voice only when that speaker covers at least
# this share of it and all other speakers together at most _EMBED_MAX_OTHER.
_EMBED_MIN_COVERAGE = 0.5
_EMBED_MAX_OTHER = 0.1
_PARAKEET_VERSION = "parakeet-mlx>=0.4,<1"
# Long windows make parakeet silently skip whole stretches of speech; 30 s windows
# cut AMI meeting WER by ~2.5 points at the same speed (eval/results/parakeet.md).
_PARAKEET_ARGS = ("--chunk-duration", "30", "--overlap-duration", "5")
_NEMOTRON_SPEC = "mlx-audio==0.5.6"
_NEMOTRON_MODEL = "mlx-community/Nemotron-3-Diarization"
_NEMOTRON_THRESHOLD = 0.5
_NEMOTRON_FRAME_S = 0.01
DIARIZERS = ("nemotron", "senko")
DEFAULT_DIARIZER = "nemotron"
_FUTURE_TIMEOUT_S = 600
_ERROR_FUTURE_TIMEOUT_S = 30


class ScribeError(Exception):
    """Raised when the transcription or diarization pipeline fails."""


class _Diarizer(Protocol):
    """Protocol for a senko-compatible diarizer."""

    def diarize(self, audio_path: str) -> dict | None: ...


class _Embedder(Protocol):
    """The senko Diarizer internals used to embed speaker turns (senko pinned commit)."""

    def _generate_subsegments(self, vad_segments: list, accurate: object) -> list: ...
    def _extract_fbank_features(self, wav_path: str, subsegments: list) -> tuple: ...
    def _generate_embeddings(self, *args: object) -> object: ...


class DiarizationSegment(NamedTuple):
    """A speaker-labeled time segment from diarization."""

    speaker: str
    start: float
    end: float


class Word(NamedTuple):
    """A single transcribed word (with any trailing punctuation) and its timestamps."""

    text: str
    start: float
    end: float


class TranscriptionSegment(NamedTuple):
    """A transcribed text segment with timestamps from parakeet.

    ``words`` holds word-level timing when parakeet provides tokens; merge()
    uses it to attribute speakers per word rather than per segment.
    """

    text: str
    start: float
    end: float
    words: tuple[Word, ...] = ()


class MergedSegment(NamedTuple):
    """A transcription segment attributed to a speaker."""

    speaker: str
    text: str
    start: float
    end: float


def _log(message: str) -> None:
    """Print a progress message to stderr."""
    print(message, file=sys.stderr)  # noqa: T201


def _find_ffmpeg() -> str:
    """Return the path to ffmpeg, or raise if not installed."""
    path = shutil.which("ffmpeg")
    if path is None:
        msg = "ffmpeg is required for audio normalization.\nInstall: brew install ffmpeg"
        raise ScribeError(msg)
    return path


def _find_uvx() -> str:
    """Return the path to uvx, or raise if not installed."""
    path = shutil.which("uvx")
    if path is None:
        msg = "uvx not found. Install uv: https://docs.astral.sh/uv/"
        raise ScribeError(msg)
    return path


@contextmanager
def _prepare_audio(
    audio_path: Path,
    *,
    on_log: Callable[[str], None] | None = _log,
) -> Iterator[Path]:
    """Yield a normalized WAV (16kHz mono), converting via ffmpeg."""
    ffmpeg = _find_ffmpeg()
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        if on_log is not None:
            on_log(f"Preparing audio ({audio_path.suffix})...")
        subprocess.run(  # noqa: S603
            [
                ffmpeg,
                "-i",
                str(audio_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(tmp_path),
                "-y",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        yield tmp_path
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "unknown error"
        msg = f"ffmpeg conversion failed: {stderr}"
        raise ScribeError(msg) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


def _create_diarizer() -> _Diarizer:
    """Create a senko Diarizer with automatic device selection."""
    try:
        import senko  # noqa: PLC0415
    except ImportError as exc:
        msg = "senko is not installed. Run: uv sync"
        raise ScribeError(msg) from exc

    try:
        return senko.Diarizer(device="auto")  # type: ignore[return-value] — no stubs
    except Exception as exc:
        msg = f"Failed to initialize diarizer: {exc}"
        raise ScribeError(msg) from exc


def run_diarization(
    diarizer: _Diarizer,
    audio_path: Path,
) -> list[DiarizationSegment]:
    """Run senko diarization on an audio file.

    Args:
        diarizer: A senko Diarizer instance.
        audio_path: Path to a 16kHz mono WAV file.
    """
    try:
        result = diarizer.diarize(str(audio_path))
        if result is None:
            return []
        return [
            DiarizationSegment(seg["speaker"], seg["start"], seg["end"])
            for seg in result["merged_segments"]
        ]
    except ScribeError:
        raise
    except Exception as exc:
        msg = f"Diarization failed on {audio_path}: {exc}"
        raise ScribeError(msg) from exc


# Runs inside the isolated mlx-audio environment. argv: wav, output json, model id,
# threshold, frame seconds. Writes [[speaker_index, start_s, end_s], ...].
_NEMOTRON_WORKER = """
import json, sys
import numpy as np
from mlx_audio.vad import load
wav, out, model_id = sys.argv[1:4]
threshold, frame_s = float(sys.argv[4]), float(sys.argv[5])
model = load(model_id, strict=True)
model.set_streaming_config("offline")
waveform, _ = model._load_audio(wav, 16000)
stream = model.generate_stream(waveform, 16000, threshold=threshold)
chunks = [np.asarray(r.speaker_probs) for r in stream]
probs = np.concatenate(chunks) if chunks else np.zeros((0, 0))
segments = []
for spk in range(probs.shape[1]):
    active = np.concatenate([[False], probs[:, spk] > threshold, [False]])
    edges = np.flatnonzero(np.diff(active.astype(np.int8)))
    for s, e in zip(edges[0::2], edges[1::2]):
        segments.append([spk, float(s * frame_s), float(e * frame_s)])
with open(out, "w") as f:
    json.dump(segments, f)
"""


def run_nemotron_diarization(audio_path: Path) -> list[DiarizationSegment]:
    """Run Nemotron 3 Diarization (mlx-audio port) on a 16 kHz mono WAV.

    The model runs in its own uvx environment, as parakeet-mlx does, so its
    dependencies never mix with scribe's. Speakers are labeled SPEAKER_00, SPEAKER_01,
    ... in order of first appearance.
    """
    uvx = _find_uvx()
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = Path(tmp_dir) / "segments.json"
        try:
            subprocess.run(  # noqa: S603
                [
                    uvx,
                    "--python",
                    "3.13",
                    "--from",
                    _NEMOTRON_SPEC,
                    "python",
                    "-c",
                    _NEMOTRON_WORKER,
                    str(audio_path),
                    str(out),
                    _NEMOTRON_MODEL,
                    str(_NEMOTRON_THRESHOLD),
                    str(_NEMOTRON_FRAME_S),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "unknown error"
            msg = f"Nemotron diarization failed: {stderr}"
            raise ScribeError(msg) from exc

        try:
            raw = json.loads(out.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            msg = f"Nemotron diarization returned no usable output: {exc}"
            raise ScribeError(msg) from exc

    return _label_by_first_appearance(raw)


def _label_by_first_appearance(raw: list[list[float]]) -> list[DiarizationSegment]:
    """Turn [speaker_index, start, end] rows into segments labeled in speaking order."""
    rows = sorted(raw, key=lambda r: (r[1], r[0]))
    labels: dict[int, str] = {}
    segments = []
    for spk, start, end in rows:
        label = labels.setdefault(int(spk), f"SPEAKER_{len(labels):02d}")
        segments.append(DiarizationSegment(label, float(start), float(end)))
    return segments


class SpeakerEmbedding(NamedTuple):
    """A speaker's CAM++ voiceprint for one recording."""

    centroid: list[float] | None
    speech_s: float
    n_subsegments: int


class AudioTranscription(NamedTuple):
    """Attributed segments and optional voiceprints, keyed by original cluster label."""

    segments: list[MergedSegment]
    embeddings: dict[str, SpeakerEmbedding]


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _covered(intervals: list[tuple[float, float]], start: float, end: float) -> float:
    """Seconds of [start, end] covered by sorted, non-overlapping intervals."""
    return sum(max(0.0, min(end, e) - max(start, s)) for s, e in intervals if s < end and e > start)


def cluster_embeddings(
    audio_path: Path,
    diarization: list[DiarizationSegment],
) -> dict[str, SpeakerEmbedding]:
    """Compute a CAM++ voiceprint per diarized speaker with senko's embedding model.

    Nemotron returns no speaker embeddings, so senko's CAM++ pipeline runs over
    the diarizer's turns instead of senko's own VAD: senko cuts them into its
    usual ~1.5 s subsegments, each subsegment is labeled with the single speaker
    covering it (overlapped or ambiguous ones are skipped), and a speaker's
    centroid is the mean of its L2-normalized embeddings. The centroid is None
    when a speaker has no usable subsegment.
    """
    turns: dict[str, list[tuple[float, float]]] = {}
    for seg in diarization:
        if seg.end > seg.start:
            turns.setdefault(seg.speaker, []).append((seg.start, seg.end))
    turns = {spk: _merge_intervals(iv) for spk, iv in turns.items()}
    speech = {spk: sum(e - s for s, e in iv) for spk, iv in turns.items()}
    if not turns:
        return {}

    embedder = _create_embedder()
    regions = _merge_intervals([iv for ivs in turns.values() for iv in ivs])
    subsegments = [(max(0.0, s), e) for s, e in embedder._generate_subsegments(regions, None)]  # noqa: SLF001

    speakers = list(turns)
    kept: list[tuple[float, float]] = []
    owners: list[str] = []
    for start, end in subsegments:
        cover = [_covered(turns[spk], start, end) / (end - start) for spk in speakers]
        best = cover.index(max(cover))
        if cover[best] >= _EMBED_MIN_COVERAGE and sum(cover) - cover[best] <= _EMBED_MAX_OTHER:
            kept.append((start, end))
            owners.append(speakers[best])

    centroids: dict[str, list[float]] = {}
    if kept:
        try:
            feats, frames, offsets, dim = embedder._extract_fbank_features(str(audio_path), kept)  # noqa: SLF001
            emb = np.asarray(
                embedder._generate_embeddings(feats, frames, [int(o) for o in offsets], dim)  # noqa: SLF001
            )
        except Exception as exc:
            msg = f"Speaker embedding failed on {audio_path}: {exc}"
            raise ScribeError(msg) from exc
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        owner_arr = np.array(owners)
        for spk in speakers:
            mask = owner_arr == spk
            if mask.any():
                centroids[spk] = [float(x) for x in emb[mask].mean(axis=0)]
    return {
        spk: SpeakerEmbedding(centroids.get(spk), speech[spk], owners.count(spk))
        for spk in speakers
    }


def _create_embedder() -> _Embedder:
    """Create a senko Diarizer used only for its CAM++ embedding pipeline."""
    try:
        import senko  # noqa: PLC0415
    except ImportError as exc:
        msg = "senko is not installed. Run: uv sync"
        raise ScribeError(msg) from exc

    try:
        return senko.Diarizer(device="auto", warmup=False, quiet=True)
    except Exception as exc:
        msg = f"Failed to initialize speaker embedder: {exc}"
        raise ScribeError(msg) from exc


def diarize(audio_path: Path, *, diarizer: str = DEFAULT_DIARIZER) -> list[DiarizationSegment]:
    """Diarize a 16 kHz mono WAV with the named diarizer (see DIARIZERS)."""
    if diarizer == "nemotron":
        return run_nemotron_diarization(audio_path)
    if diarizer == "senko":
        return run_diarization(_create_diarizer(), audio_path)
    msg = f"Unknown diarizer {diarizer!r}; choose one of {', '.join(DIARIZERS)}"
    raise ScribeError(msg)


def run_transcription(
    audio_path: Path,
) -> list[TranscriptionSegment]:
    """Run parakeet-mlx transcription on an audio file."""
    uvx = _find_uvx()
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            subprocess.run(  # noqa: S603
                [
                    uvx,
                    _PARAKEET_VERSION,
                    str(audio_path),
                    "--output-format",
                    "json",
                    "--output-dir",
                    tmp_dir,
                    *_PARAKEET_ARGS,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "unknown error"
            msg = f"parakeet-mlx failed: {stderr}"
            raise ScribeError(msg) from exc

        json_files = list(Path(tmp_dir).glob("*.json"))
        if not json_files:
            msg = "parakeet-mlx produced no JSON output"
            raise ScribeError(msg)

        try:
            data = json.loads(json_files[0].read_text())
        except json.JSONDecodeError as exc:
            msg = f"parakeet-mlx returned invalid JSON: {exc}"
            raise ScribeError(msg) from exc

    segments = []
    for segment in data.get("segments", data.get("sentences", [])):
        text = segment.get("text", "").strip()
        start = segment.get("start")
        end = segment.get("end")
        if text and start is not None and end is not None:
            words = _words_from_tokens(segment.get("tokens", []))
            segments.append(TranscriptionSegment(text, float(start), float(end), words))
    return segments


def _is_punctuation(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and all(c in _PUNCTUATION for c in stripped)


def _words_from_tokens(tokens: list[dict]) -> tuple[Word, ...]:
    """Group parakeet subword tokens into words.

    A token with a leading space starts a new word; any other token (a word
    piece) extends the current one. Punctuation always attaches to the word
    before it without extending its end time, since parakeet stamps it later
    than the speech it closes.
    """
    words: list[Word] = []
    for token in tokens:
        text = token.get("text", "")
        start = token.get("start")
        end = token.get("end")
        if not text or start is None or end is None:
            continue
        punctuation = _is_punctuation(text)
        if words and (punctuation or not text.startswith(" ")):
            prev = words[-1]
            new_end = prev.end if punctuation else float(end)
            words[-1] = Word(prev.text + text.strip(), prev.start, new_end)
        elif not punctuation:
            words.append(Word(text.strip(), float(start), float(end)))
    return tuple(w for w in words if w.text)


def _overlap_speaker(
    diarization: list[DiarizationSegment],
    d_start: int,
    start: float,
    end: float,
) -> str | None:
    """Return the speaker with the most overlap in [start, end], or None.

    diarization[d_start] must be the first segment ending after ``start``.
    """
    best_speaker = None
    best_overlap = 0.0
    for i in range(d_start, len(diarization)):
        dseg = diarization[i]
        if dseg.start >= end:
            break
        overlap = min(end, dseg.end) - max(start, dseg.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = dseg.speaker
    return best_speaker


def _nearest_speaker(
    diarization: list[DiarizationSegment],
    d_start: int,
    start: float,
    end: float,
) -> str:
    """Return the speaker of the nearest segment within the tolerance, else UNKNOWN."""
    best_speaker = UNKNOWN_SPEAKER
    best_gap = _NEAREST_SPEAKER_TOLERANCE_S
    for i in (d_start - 1, d_start):
        if 0 <= i < len(diarization):
            dseg = diarization[i]
            gap = max(dseg.start - end, start - dseg.end, 0.0)
            if gap <= best_gap:
                best_gap = gap
                best_speaker = dseg.speaker
    return best_speaker


def merge(
    diarization: list[DiarizationSegment],
    transcription: list[TranscriptionSegment],
) -> list[MergedSegment]:
    """Assign speakers to transcription by time overlap, word by word.

    Each word takes the speaker it overlaps most, and a segment is split
    wherever the speaker changes between words. A word that overlaps no
    speaker keeps the speaker of the word before it in the same segment; only
    a segment's first word falls back to the nearest speaker. Short runs that
    are likely diarization jitter are then smoothed (see _smooth_runs). A
    segment without word timings is attributed as a whole.
    """
    diarization = sorted(diarization, key=lambda s: s.start)
    transcription = sorted(transcription, key=lambda s: s.start)
    if not diarization:
        return [MergedSegment(UNKNOWN_SPEAKER, s.text, s.start, s.end) for s in transcription]
    merged: list[MergedSegment] = []
    d_start = 0
    for seg in transcription:
        words = seg.words or (Word(seg.text, seg.start, seg.end),)
        runs: list[tuple[str, list[Word]]] = []
        for word in words:
            while d_start < len(diarization) and diarization[d_start].end <= word.start:
                d_start += 1
            speaker = _overlap_speaker(diarization, d_start, word.start, word.end)
            if speaker is None:
                speaker = (
                    runs[-1][0]
                    if runs
                    else _nearest_speaker(diarization, d_start, word.start, word.end)
                )
            if runs and runs[-1][0] == speaker:
                runs[-1][1].append(word)
            else:
                runs.append((speaker, [word]))
        merged.extend(_merged_from_run(speaker, run) for speaker, run in _smooth_runs(runs))
    return merged


def _smooth_runs(runs: list[tuple[str, list[Word]]]) -> list[tuple[str, list[Word]]]:
    """Relabel short speaker runs inside one segment that are likely diarization jitter.

    A run of at most _SMOOTH_MAX_WORDS words takes its neighbours' speaker when
    it sits between two runs of the same speaker, or when it is the segment's
    first or last run and no pause of _SMOOTH_MIN_PAUSE_S separates it from
    the next or previous run. Tuned on AMI (eval/merge_refine.py): cuts
    speaker-attributed WER by ~0.8 points and spurious speaker switches by ~14%.
    """
    runs = [(speaker, list(words)) for speaker, words in runs]
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, (speaker, words) in enumerate(runs):
            if len(words) > _SMOOTH_MAX_WORDS:
                continue
            prev = runs[i - 1] if i > 0 else None
            nxt = runs[i + 1] if i + 1 < len(runs) else None
            target = None
            if prev and nxt and prev[0] == nxt[0]:
                target = prev[0]
            elif prev is None and nxt and nxt[1][0].start - words[-1].end < _SMOOTH_MIN_PAUSE_S:
                target = nxt[0]
            elif nxt is None and prev and words[0].start - prev[1][-1].end < _SMOOTH_MIN_PAUSE_S:
                target = prev[0]
            if target is not None and target != speaker:
                runs[i] = (target, words)
                runs = _join_adjacent(runs)
                changed = True
                break
    return runs


def _join_adjacent(runs: list[tuple[str, list[Word]]]) -> list[tuple[str, list[Word]]]:
    joined: list[tuple[str, list[Word]]] = []
    for speaker, words in runs:
        if joined and joined[-1][0] == speaker:
            joined[-1][1].extend(words)
        else:
            joined.append((speaker, list(words)))
    return joined


def _merged_from_run(speaker: str, words: list[Word]) -> MergedSegment:
    """Build a MergedSegment from consecutive words by one speaker."""
    text = " ".join(w.text for w in words)
    return MergedSegment(speaker, text, words[0].start, words[-1].end)


def format_text(segments: list[MergedSegment]) -> str:
    """Format as plain text, merging consecutive same-speaker segments."""
    if not segments:
        return ""

    lines: list[str] = []
    current_speaker: str | None = None
    current_texts: list[str] = []

    for seg in segments:
        if seg.speaker != current_speaker:
            if current_texts:
                lines.append(f"{current_speaker}: {' '.join(current_texts)}")
                lines.append("")
            current_speaker = seg.speaker
            current_texts = [seg.text]
        else:
            current_texts.append(seg.text)

    if current_texts:
        lines.append(f"{current_speaker}: {' '.join(current_texts)}")

    return "\n".join(lines).strip() + "\n"


def format_json(segments: list[MergedSegment]) -> str:
    """Format as JSON with timestamps and speaker attribution."""
    speakers = sorted({seg.speaker for seg in segments})
    data = {
        "speakers": speakers,
        "segments": [
            {
                "speaker": seg.speaker,
                "text": seg.text,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
            }
            for seg in segments
        ],
    }
    return json.dumps(data, indent=2) + "\n"


def format_transcript_text(segments: list[TranscriptionSegment]) -> str:
    """Format transcription as plain text without speaker labels."""
    if not segments:
        return ""
    return " ".join(seg.text for seg in segments) + "\n"


def format_transcript_json(segments: list[TranscriptionSegment]) -> str:
    """Format transcription as JSON without speaker attribution."""
    data = {
        "segments": [
            {
                "text": seg.text,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
            }
            for seg in segments
        ],
    }
    return json.dumps(data, indent=2) + "\n"


def _log_secondary_transcription_error(
    future: Future[list[TranscriptionSegment]], on_log: Callable[[str], None],
) -> None:
    """Collect standalone diagnostics without replacing the primary diarization error."""
    try:
        future.result(timeout=_ERROR_FUTURE_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001
        on_log(f"Transcription also failed: {exc}")


def transcribe_audio(
    original_path: Path,
    *,
    diarizer: str = DEFAULT_DIARIZER,
    include_embeddings: bool = False,
    on_step: Callable[[int, int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> AudioTranscription:
    """Prepare audio, recognize and diarize concurrently, then attribute words.

    ``on_step(index, 5)`` announces phase starts: preparation (0), ASR launch
    (1), diarization (2), optional embeddings (3), and merge (4). A phase start
    is not a completion notification; only a returned result means success.

    The normalized WAV stays alive until the ASR worker has stopped, even on
    failure. ASR collection has a 600-second timeout, but executor shutdown still
    waits for the worker. With ``on_log``, a diarization failure also collects a
    secondary ASR diagnostic for up to 30 seconds without replacing that failure.
    Embedding failures simply unwind and wait. No output files are persisted.
    """
    def step(index: int) -> None:
        if on_step is not None:
            on_step(index, 5)

    def log(message: str) -> None:
        if on_log is not None:
            on_log(message)

    step(0)
    with _prepare_audio(original_path, on_log=on_log) as audio_path:
        with ThreadPoolExecutor(max_workers=1) as executor:
            step(1)
            log("Starting transcription (background)...")
            transcription_future = executor.submit(run_transcription, audio_path)

            step(2)
            try:
                log(f"Running {diarizer} diarization on {audio_path}...")
                diarization = diarize(audio_path, diarizer=diarizer)
            except Exception:
                if on_log is not None:
                    _log_secondary_transcription_error(transcription_future, on_log)
                raise

            speaker_count = len({s.speaker for s in diarization})
            log(f"  Found {speaker_count} speakers")
            if not diarization:
                log("Warning: no speech segments detected")

            embeddings: dict[str, SpeakerEmbedding] = {}
            if include_embeddings:
                step(3)
                embeddings = cluster_embeddings(audio_path, diarization)

            transcription = transcription_future.result(timeout=_FUTURE_TIMEOUT_S)
            log(f"  Transcribed {len(transcription)} segments")

        step(4)
        log("Merging results...")
        merged = merge(diarization, transcription)

        unknown_count = sum(s.speaker == UNKNOWN_SPEAKER for s in merged)
        if unknown_count:
            log(f"Warning: {unknown_count} segments could not be attributed to a speaker")

    return AudioTranscription(merged, embeddings)


def _run_transcript_only(original_path: Path, *, output_format: str) -> str:
    """Prepare and transcribe original audio without diarization."""
    with _prepare_audio(original_path) as audio_path:
        _log("Running transcription...")
        transcription = run_transcription(audio_path)
        _log(f"  Transcribed {len(transcription)} segments")
    return (
        format_transcript_json(transcription)
        if output_format == "json"
        else format_transcript_text(transcription)
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Local transcription and speaker diarization with parakeet + Nemotron 3",
    )
    parser.add_argument("audio", type=Path, help="Path to audio file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path, or - for stdout (default: input file with .txt/.json extension)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        dest="output_format",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Transcribe only, without speaker diarization",
    )
    parser.add_argument(
        "--diarizer",
        choices=DIARIZERS,
        default=DEFAULT_DIARIZER,
        help=f"Speaker diarization model (default: {DEFAULT_DIARIZER})",
    )
    args = parser.parse_args()

    if not args.audio.exists():
        sys.exit(f"Error: audio file not found: {args.audio}")

    try:
        if args.no_diarize:
            output = _run_transcript_only(
                args.audio,
                output_format=args.output_format,
            )
        else:
            result = transcribe_audio(
                args.audio,
                diarizer=args.diarizer,
                on_log=_log,
            )
            output = (
                format_json(result.segments)
                if args.output_format == "json"
                else format_text(result.segments)
            )

        if args.output == "-":
            sys.stdout.write(output)
        else:
            ext = ".json" if args.output_format == "json" else ".txt"
            output_path = Path(args.output) if args.output else args.audio.with_suffix(ext)
            output_path.write_text(output)
            _log(f"Wrote transcript to {output_path}")

    except ScribeError as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
