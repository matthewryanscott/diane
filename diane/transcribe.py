from typing import NamedTuple

from scribe import MergedSegment, transcribe_audio

from diane.db import RECORDINGS_DIR, Recording
from diane.voiceprints import Cluster


def _format_timestamp(seconds: float) -> str:
    """Format seconds as [HH:MM:SS] or [MM:SS]."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"[{h:02d}:{m:02d}:{s:02d}]"
    return f"[{m:02d}:{s:02d}]"


PAUSE_THRESHOLD_S = 2.0
MAX_DURATION_S = 120.0


def _should_break(
    current_texts: list[str],
    current_start: float,
    prev_end: float,
    next_start: float,
) -> bool:
    """Decide whether to break within a speaker's turn.

    Breaks on pause gaps or when the current block exceeds the max duration,
    but only at sentence boundaries.
    """
    has_pause = (next_start - prev_end) >= PAUSE_THRESHOLD_S
    over_duration = (prev_end - current_start) >= MAX_DURATION_S

    if not (has_pause or over_duration):
        return False

    # Only break if the last segment ends with sentence-ending punctuation
    last_text = current_texts[-1].rstrip()
    return last_text[-1:] in ".!?…" if last_text else False


def format_diarized_transcript(segments: list[MergedSegment]) -> str:
    """Format merged segments as timestamped, diarized text."""
    if not segments:
        return ""

    lines: list[str] = []
    current_speaker: str | None = None
    current_texts: list[str] = []
    current_start: float = 0.0
    prev_end: float = 0.0

    def _flush() -> None:
        if current_texts:
            ts = _format_timestamp(current_start)
            lines.append(f"{ts} {current_speaker}: {' '.join(current_texts)}")
            lines.append("")

    for seg in segments:
        if seg.speaker != current_speaker:
            _flush()
            current_speaker = seg.speaker
            current_texts = [seg.text]
            current_start = seg.start
        elif _should_break(current_texts, current_start, prev_end, seg.start):
            _flush()
            current_texts = [seg.text]
            current_start = seg.start
        else:
            current_texts.append(seg.text)
        prev_end = seg.end

    _flush()

    return "\n".join(lines).strip() + "\n"


STEPS = [
    "Preparing audio",
    "Transcribing",
    "Diarizing",
    "Speaker voiceprints",
    "Merging",
]


class TranscriptionResult(NamedTuple):
    """Speaker-attributed segments plus a voiceprint per speaker cluster."""

    segments: list[MergedSegment]
    clusters: list[Cluster]


def transcribe_recording(
    recording: Recording,
    *,
    on_step: callable = lambda step, total: None,
) -> TranscriptionResult:
    """Transcribe and diarize a recording, and embed each speaker cluster.

    on_step is called with (step_index, total_steps) as each phase begins.
    """
    result = transcribe_audio(
        RECORDINGS_DIR / recording.path,
        include_embeddings=True,
        on_step=on_step,
    )
    clusters = [
        Cluster(speaker, embedding.centroid, round(embedding.speech_s, 2))
        for speaker, embedding in result.embeddings.items()
        if embedding.centroid is not None
    ]
    return TranscriptionResult(result.segments, clusters)
