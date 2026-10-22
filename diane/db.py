import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

# Capture the CLI's startup workspace, independent of the package install location.
PROJECT_ROOT = Path.cwd()
RECORDINGS_DIR = PROJECT_ROOT / "Recordings"

# Matches: YYYYMMDD HHMM description.ext
_FILENAME_RE = re.compile(r"^(\d{8}) (\d{4}) (.+)$")
# Matches: description YYYYMMDD HHMM.ext
_TITLE_FIRST_FILENAME_RE = re.compile(r"^(.+) (\d{8}) (\d{4})$")


@dataclass
class Recording:
    unique_id: str
    date: datetime
    duration: timedelta
    title: str
    path: str

    @property
    def short_id(self) -> str:
        return self.unique_id[:8]


@dataclass
class Transcript:
    """A transcript's filename association, optionally enriched by recording discovery."""

    path: Path
    filename_id: str
    recording: Recording | None = None


def _ffprobe_duration(filepath: Path) -> float:
    """Get audio duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "json", str(filepath)],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except Exception:  # noqa: BLE001
        return 0.0


def _birthtime(filepath: Path) -> datetime | None:
    """Get file creation time (birthtime) with second precision. Returns None on failure."""
    try:
        stat = os.stat(filepath)
        # st_birthtime available on macOS
        return datetime.fromtimestamp(stat.st_birthtime).astimezone()
    except (AttributeError, OSError):
        return None


def get_recordings() -> list[Recording]:
    """Discover date-first or title-first recordings, newest first, without changing sources."""
    if not RECORDINGS_DIR.exists():
        return []

    # First pass: parse filenames to find matching files
    candidates: list[tuple[Path, str, datetime, str]] = []
    for f in RECORDINGS_DIR.iterdir():
        if not f.is_file() or f.name.startswith("."):
            continue
        m = _FILENAME_RE.match(f.stem)
        if m:
            date_str, time_str, title = m.groups()
        else:
            m = _TITLE_FIRST_FILENAME_RE.match(f.stem)
            if not m:
                continue
            title, date_str, time_str = m.groups()
        # Prefer birthtime (second precision) over filename (minute precision)
        dt = _birthtime(f)
        if dt is None:
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M").astimezone()
        unique_id = hashlib.sha256(f.stem.encode()).hexdigest()
        candidates.append((f, unique_id, dt, title))

    # Second pass: get durations via ffprobe
    recordings = []
    for filepath, unique_id, dt, title in candidates:
        duration = timedelta(seconds=_ffprobe_duration(filepath))
        recordings.append(Recording(
            unique_id=unique_id,
            date=dt,
            duration=duration,
            title=title,
            path=filepath.name,
        ))

    recordings.sort(key=lambda r: r.date, reverse=True)
    return recordings


def find_recording(short_id: str) -> Recording:
    """Find a recording by short ID prefix. Raises if not found or ambiguous."""
    short_id = short_id.lower()
    matches = [r for r in get_recordings() if r.short_id.startswith(short_id)]
    if not matches:
        msg = f"No recording found matching '{short_id}'"
        raise ValueError(msg)
    if len(matches) > 1:
        msg = f"Ambiguous ID '{short_id}' matches {len(matches)} recordings"
        raise ValueError(msg)
    return matches[0]


def transcript_id(
    path: Path, *, policy: Literal["suffix-or-stem", "delimited-only"] = "suffix-or-stem"
) -> str:
    """Parse the ID after the first hyphen, preserving case and legacy undelimited names.

    ``suffix-or-stem`` uses the whole stem when no hyphen exists; ``delimited-only``
    returns an empty association ID in that case. Neither policy reads frontmatter.
    """
    stem = path.stem
    if "-" in stem:
        return stem.split("-", 1)[-1]
    return "" if policy == "delimited-only" else stem


def canonical_transcript_path(recording: Recording, directory: Path) -> Path:
    """Return the exact single-transcription path, not an alternate same-ID filename."""
    date_str = (
        recording.date.astimezone().strftime("%Y%m%dT%H%M%S") if recording.date else "unknown"
    )
    return directory / f"{date_str}-{recording.short_id}.txt"


def list_transcripts(
    directory: Path,
    *,
    id_policy: Literal["suffix-or-stem", "delimited-only"] = "suffix-or-stem",
    include_recordings: bool = False,
) -> list[Transcript]:
    """Enumerate direct ``*.txt`` entries without sorting.

    By default, recording discovery never runs. When requested, enrichment discovers
    recordings only after finding transcript entries and joins exact, case-sensitive
    filename IDs under the requested policy. Orphans remain in the result. A missing
    directory is empty; other directory I/O errors propagate.
    """
    transcripts = []
    try:
        for path in directory.iterdir():
            if not path.name.endswith(".txt"):
                continue
            filename_id = transcript_id(path, policy=id_policy)
            transcripts.append(Transcript(path, filename_id))
    except FileNotFoundError:
        return []
    if transcripts and include_recordings:
        by_id = {recording.short_id: recording for recording in get_recordings()}
        for transcript in transcripts:
            transcript.recording = by_id.get(transcript.filename_id)
    return transcripts


def find_transcript(prefix: str, directory: Path) -> Transcript:
    """Resolve a case-insensitive filename-ID prefix without recording discovery."""
    prefix = prefix.lower()
    matches = [
        transcript for transcript in list_transcripts(directory)
        if transcript.filename_id.lower().startswith(prefix)
    ]
    if not matches:
        msg = f"No transcript for {prefix}."
        raise FileNotFoundError(msg)
    if len(matches) > 1:
        msg = f"Ambiguous ID '{prefix}' matches {len(matches)} transcripts"
        raise ValueError(msg)
    return matches[0]


def find_recording_transcript(prefix: str, directory: Path) -> Transcript:
    """Resolve recording metadata first, then require its exact canonical transcript."""
    recording = find_recording(prefix)
    path = canonical_transcript_path(recording, directory)
    if not path.exists():
        msg = f"No transcript for {prefix}. Run 'diane transcribe {prefix}' first."
        raise FileNotFoundError(msg)
    return Transcript(path, transcript_id(path), recording)


def transcribed_ids(directory: Path) -> set[str]:
    """Return exact suffix-or-stem IDs for bulk skip/status, regardless of canonical path."""
    return {transcript.filename_id for transcript in list_transcripts(directory)}


def pending_recordings(directory: Path) -> list[Recording]:
    """Discover newest-first recordings with source audio and no exact filename-ID match."""
    recordings = get_recordings()
    already = transcribed_ids(directory)
    return [
        recording for recording in recordings
        if recording.short_id not in already and (RECORDINGS_DIR / recording.path).exists()
    ]
