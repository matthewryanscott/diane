import re
from datetime import datetime, timedelta

from diane.speakers import parse_timestamp

# Matches relative timestamps: [MM:SS] or [HH:MM:SS]
_REL_TIMESTAMP_RE = re.compile(r"\[(\d+(?::\d{2}){1,2})\]")

# Matches absolute timestamps: [2026-04-02T10:05:00]
_ABS_TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[^\]]*)\]", re.MULTILINE)


def is_absolute(frontmatter: dict) -> bool:
    return frontmatter.get("timestamps") == "absolute"


def to_absolute(body: str, recording_start: datetime) -> str:
    """Replace all relative [MM:SS] or [HH:MM:SS] timestamps with absolute ISO timestamps."""

    def _replace(match: re.Match) -> str:
        offset = parse_timestamp(match.group(1))
        absolute = recording_start + timedelta(seconds=offset)
        return f"[{absolute.strftime('%Y-%m-%dT%H:%M:%S')}]"

    return _REL_TIMESTAMP_RE.sub(_replace, body)


def extract_timestamp(line: str) -> datetime | None:
    """Extract absolute timestamp from a line, or None."""
    match = _ABS_TIMESTAMP_RE.match(line)
    if match:
        return datetime.fromisoformat(match.group(1))
    return None


def timestamp_line_content(line: str) -> str | None:
    """Return content after the timestamp (stripped), or None if line has no timestamp."""
    match = _ABS_TIMESTAMP_RE.match(line)
    if not match:
        return None
    return line[match.end():].strip()


def shift_timestamps(body: str, delta: timedelta) -> str:
    """Shift all absolute ISO timestamps in body by delta."""

    def _shift(match: re.Match) -> str:
        ts = datetime.fromisoformat(match.group(1))
        shifted = ts + delta
        return f"[{shifted.strftime('%Y-%m-%dT%H:%M:%S')}]"

    return _ABS_TIMESTAMP_RE.sub(_shift, body)


def slice_transcript(body: str, start: datetime, end: datetime) -> str:
    """Return lines whose absolute timestamps fall within [start, end]."""
    # Normalize to naive for comparison (transcript timestamps are local, no tz info)
    start_naive = start.replace(tzinfo=None) if start.tzinfo else start
    end_naive = end.replace(tzinfo=None) if end.tzinfo else end
    lines = []
    for line in body.splitlines():
        match = _ABS_TIMESTAMP_RE.match(line)
        if match:
            ts = datetime.fromisoformat(match.group(1))
            ts_naive = ts.replace(tzinfo=None) if ts.tzinfo else ts
            if start_naive <= ts_naive <= end_naive:
                lines.append(line)
        # Include non-timestamped lines (like blank lines) if they follow a matching line
        elif lines and not line.strip():
            lines.append(line)
    return "\n".join(lines) + "\n" if lines else ""
