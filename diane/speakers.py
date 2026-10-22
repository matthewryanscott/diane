import re
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

from rich.console import Console

# Matches transcript lines with relative or absolute timestamps:
#   [MM:SS] SPEAKER: text
#   [HH:MM:SS] SPEAKER: text
#   [2026-04-02T16:59:20] SPEAKER: text
#   [2026-04-02T16:59:20] Synthetic Speaker: text   (names may contain spaces)
_SEGMENT_RE = re.compile(r"^\[([^\]]+)\] ([^\s:]+(?: [^\s:]+)*): (.*)$", re.MULTILINE)


_UNNAMED_RE = re.compile(r"^SPEAKER_\d+$")


def unnamed_speakers(body: str) -> list[str]:
    """Return the diarizer labels (SPEAKER_00, ...) that no name has replaced yet."""
    return [s for s in extract_speakers(body) if _UNNAMED_RE.match(s)]


def speaker_labels(body: str) -> list[str]:
    """Return speaker-label occurrences in transcript order, including repeats."""
    return [match.group(2) for match in _SEGMENT_RE.finditer(body)]


def extract_speakers(body: str) -> list[str]:
    """Return unique speaker labels in order of first appearance."""
    seen: set[str] = set()
    result: list[str] = []
    for match in _SEGMENT_RE.finditer(body):
        speaker = match.group(2)
        if speaker not in seen:
            seen.add(speaker)
            result.append(speaker)
    return result


def parse_timestamp(ts: str) -> float:
    """Parse MM:SS, HH:MM:SS, or absolute ISO timestamp to seconds offset.

    For absolute timestamps, returns seconds since midnight as an approximation
    (used only for relative ordering within a transcript).
    """
    if "T" in ts:
        # Absolute: 2026-04-02T16:59:20
        time_part = ts.split("T", 1)[1]
        parts = time_part.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    parts = ts.split(":")
    if len(parts) == 2:  # noqa: PLR2004
        return int(parts[0]) * 60 + int(parts[1])
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def _all_segment_times(body: str, rec_start_seconds: float | None = None) -> list[tuple[float, str]]:
    """Return all (start_seconds, speaker) pairs as offsets from recording start.

    For absolute timestamps, subtracts recording start time (seconds-since-midnight)
    to get offset into audio file. If rec_start_seconds is None, falls back to
    first segment as base (less accurate).
    For relative timestamps, values are already relative to recording start.
    """
    raw = [
        (parse_timestamp(m.group(1)), m.group(2))
        for m in _SEGMENT_RE.finditer(body)
    ]
    if not raw:
        return raw
    first_match = _SEGMENT_RE.search(body)
    if first_match and "T" in first_match.group(1):
        base = rec_start_seconds if rec_start_seconds is not None else raw[0][0]
        return [(t - base, spk) for t, spk in raw]
    return raw


def extract_speaker_segments(
    body: str, speaker: str, *, rec_start_seconds: float | None = None,
) -> list[tuple[float, float]]:
    """Return list of (start, end) seconds for segments belonging to speaker.

    End time is estimated from the next segment's start, or +10s for the last.
    rec_start_seconds: for absolute transcripts, the recording start as seconds-since-midnight.
    """
    all_times = _all_segment_times(body, rec_start_seconds)
    segments: list[tuple[float, float]] = []
    for i, (start, spk) in enumerate(all_times):
        if spk != speaker:
            continue
        end = all_times[i + 1][0] if i + 1 < len(all_times) else start + 10.0
        if end > start:
            segments.append((start, end))
    return segments


_SNIPPET_MAX = 15.0


def _cap_snippet(start: float, end: float) -> tuple[float, float]:
    """Cap a snippet to _SNIPPET_MAX duration."""
    if end - start > _SNIPPET_MAX:
        end = start + _SNIPPET_MAX
    return (start, end)


def get_snippets(
    segments: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return multiple snippets sorted longest-first, each capped at 15s."""
    if not segments:
        return [(0.0, _SNIPPET_MAX)]
    capped = [_cap_snippet(s, e) for s, e in segments]
    seen: set[tuple[float, float]] = set()
    unique: list[tuple[float, float]] = []
    for snippet in sorted(capped, key=lambda s: s[1] - s[0], reverse=True):
        if snippet not in seen:
            seen.add(snippet)
            unique.append(snippet)
    return unique


def first_excerpt(body: str, speaker: str, max_chars: int = 100) -> str:
    """Return the first bit of text from a speaker."""
    for match in _SEGMENT_RE.finditer(body):
        if match.group(2) == speaker:
            text = match.group(3)
            if len(text) > max_chars:
                return text[:max_chars] + "..."
            return text
    return ""


def _speaker_texts(
    body: str, speaker: str, rec_start_seconds: float | None = None,
) -> list[tuple[float, str]]:
    """Return (time_offset, text) pairs for a speaker, using same timing as _all_segment_times."""
    all_times = _all_segment_times(body, rec_start_seconds)
    # Build parallel list of texts from regex matches
    all_matches = list(_SEGMENT_RE.finditer(body))
    result: list[tuple[float, str]] = []
    for (t, spk), m in zip(all_times, all_matches):
        if spk == speaker:
            result.append((t, m.group(3)))
    return result


def _is_absolute_body(body: str) -> bool:
    """Check if transcript body uses absolute ISO timestamps."""
    first_match = _SEGMENT_RE.search(body)
    return bool(first_match and "T" in first_match.group(1))


def _raw_timestamp_for_offset(
    body: str, offset: float, rec_start_seconds: float | None = None,
) -> str | None:
    """Find the raw timestamp string closest to a given offset.

    For absolute transcripts, returns the ISO timestamp.
    For relative transcripts, returns None (offset already is the display value).
    """
    if not _is_absolute_body(body):
        return None
    all_times = _all_segment_times(body, rec_start_seconds)
    all_matches = list(_SEGMENT_RE.finditer(body))
    best_match = None
    best_dist = float("inf")
    for (t, _spk), m in zip(all_times, all_matches):
        dist = abs(t - offset)
        if dist < best_dist:
            best_dist = dist
            best_match = m.group(1)
    return best_match


def snippet_text(
    body: str,
    speaker: str,
    snippet: tuple[float, float],
    max_chars: int = 120,
    rec_start_seconds: float | None = None,
) -> str:
    """Return transcript text for a speaker within a snippet's time range."""
    s_start, s_end = snippet
    texts = [
        text for t, text in _speaker_texts(body, speaker, rec_start_seconds)
        if t >= s_start and t < s_end
    ]
    combined = " ".join(texts)
    if len(combined) > max_chars:
        return combined[:max_chars] + "..."
    return combined


def play_loop(audio_path: Path, start: float, duration: float) -> subprocess.Popen:
    """Start looping playback of an audio snippet. Returns the process handle."""
    return subprocess.Popen(  # noqa: S603, S607
        [
            "ffplay",
            "-loop", "0",
            "-ss", str(start),
            "-t", str(duration),
            "-nodisp",
            "-loglevel", "quiet",
            str(audio_path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_playback(proc: subprocess.Popen) -> None:
    proc.terminate()
    proc.wait()


def _render_progress_bar(elapsed: float, duration: float, width: int = 20) -> str:
    """Render a text progress bar showing playback position within the snippet."""
    # Loop: elapsed wraps around the duration
    position = elapsed % duration if duration > 0 else 0
    frac = position / duration if duration > 0 else 0
    filled = int(width * frac)
    return "▓" * filled + "░" * (width - filled)


def prompt_with_snippets(
    speaker: str,
    snippets: list[tuple[float, float]],
    audio_path: Path,
    console: Console,
    *,
    body: str = "",
    rec_start_seconds: float | None = None,
) -> str:
    """Prompt for a speaker name while playing audio. Tab cycles through snippets."""
    snippet_idx = 0
    start, end = snippets[0]
    duration = end - start
    proc = play_loop(audio_path, start, duration)
    playback_start = time.monotonic()

    is_abs = _is_absolute_body(body) if body else False

    def _snippet_info_str(idx: int) -> str:
        s, e = snippets[idx]
        total = len(snippets)
        label = f"({idx + 1}/{total})" if total > 1 else ""
        time_range = f"{_format_time(s)}\u2013{_format_time(e)}"
        lines = [
            f"  [dim]♪ Playing audio {label}"
            f" ({time_range})[/dim]",
        ]
        # Show absolute timestamps for debugging alignment
        if is_abs and body:
            abs_start = _raw_timestamp_for_offset(body, s, rec_start_seconds)
            abs_end = _raw_timestamp_for_offset(body, e, rec_start_seconds)
            if abs_start:
                abs_range = f"{abs_start}\u2013{abs_end}" if abs_end else abs_start
                lines.append(f"  [dim]transcript: {abs_range}[/dim]")
                lines.append(f"  [dim]ffplay -ss {s:.1f} -t {e - s:.1f}[/dim]")
        text = snippet_text(body, speaker, snippets[idx], rec_start_seconds=rec_start_seconds) if body else ""
        if text:
            lines.append(f'  [dim]"{text}"[/dim]')
        return "\n".join(lines)

    def _progress_line() -> str:
        elapsed = time.monotonic() - playback_start
        bar = _render_progress_bar(elapsed, duration)
        pos = elapsed % duration if duration > 0 else 0
        return f"  [dim]{bar} {pos:.0f}s/{duration:.0f}s[/dim]"

    console.print(_snippet_info_str(0))

    tab_hint = " [dim](Tab=next snippet)[/dim]" if len(snippets) > 1 else ""
    prompt_prefix = f"  Name for [cyan]{speaker}[/cyan]:{tab_hint} "

    buf: list[str] = []
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)

        # Print initial progress + prompt
        sys.stdout.write("\r\n")
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        console.print(_progress_line())
        console.print(prompt_prefix, end="")
        tty.setraw(fd)

        import select

        while True:
            # Poll stdin with timeout for progress updates
            ready, _, _ = select.select([sys.stdin], [], [], 0.25)
            if ready:
                ch = sys.stdin.read(1)
                if ch == "\t" and len(snippets) > 1:
                    stop_playback(proc)
                    snippet_idx = (snippet_idx + 1) % len(snippets)
                    start, end = snippets[snippet_idx]
                    duration = end - start
                    proc = play_loop(audio_path, start, duration)
                    playback_start = time.monotonic()
                    sys.stdout.write("\r\n")
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    console.print(_snippet_info_str(snippet_idx))
                    console.print(_progress_line())
                    console.print(
                        f"  Name for [cyan]{speaker}[/cyan]: [green]{''.join(buf)}[/green]",
                        end="",
                    )
                    tty.setraw(fd)
                elif ch in ("\r", "\n"):
                    sys.stdout.write("\r\n")
                    break
                elif ch in ("\x7f", "\x08"):  # backspace
                    if buf:
                        buf.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                elif ch == "\x03":  # Ctrl-C
                    sys.stdout.write("\r\n")
                    raise KeyboardInterrupt
                elif ch >= " ":  # printable
                    buf.append(ch)
                    sys.stdout.write(ch)
                    sys.stdout.flush()
            else:
                # No input — update progress bar in-place
                # Save cursor, move up to progress line, overwrite, restore
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                # Move up 1 line, overwrite progress, move back down
                sys.stdout.write("\0337")  # save cursor
                sys.stdout.write("\033[A")  # move up (to progress line)
                sys.stdout.write("\r")  # beginning of line
                sys.stdout.write("\033[2K")  # clear line
                sys.stdout.flush()
                console.print(_progress_line(), end="")
                sys.stdout.write("\0338")  # restore cursor
                sys.stdout.flush()
                tty.setraw(fd)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        stop_playback(proc)

    return "".join(buf).strip()


def replace_speakers(body: str, mapping: dict[str, str]) -> str:
    """Replace labels simultaneously, without cascading names through the mapping."""
    return _SEGMENT_RE.sub(
        lambda match: (
            f"[{match.group(1)}] {mapping.get(match.group(2), match.group(2))}: {match.group(3)}"
        ),
        body,
    )


def _format_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS for display."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
