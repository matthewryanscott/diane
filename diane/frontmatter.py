from pathlib import Path

import yaml

from diane.db import Recording, transcript_id


def parse_frontmatter(text: str) -> tuple[dict | None, str]:
    """Split transcript into (frontmatter_dict, body). Returns (None, text) if no frontmatter."""
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return None, text
    fm_text = text[4:end]
    body = text[end + 5 :]
    return yaml.safe_load(fm_text) or {}, body


def read_frontmatter_head(path: Path) -> dict | None:
    """Parse a transcript's front matter reading only the head of the file.

    Reads line by line until the closing delimiter, so large transcript
    bodies are never loaded. Returns None if the file has no front matter.
    """
    with path.open() as f:
        if f.readline() != "---\n":
            return None
        lines: list[str] = []
        for line in f:
            if line == "---\n":
                return yaml.safe_load("".join(lines)) or {}
            lines.append(line)
    return None


def transcript_record(path: Path) -> dict:
    """One `transcript list --json` line: the transcript's front matter, keyed for scripts."""
    fm = read_frontmatter_head(path) or {}
    recorded = fm.get("recorded")
    duration = fm.get("duration")
    return {
        "id": fm.get("recording_id") or transcript_id(path),
        "recorded": str(recorded) if recorded is not None else None,
        "duration_seconds": float(duration) if duration is not None else None,
        "title": fm.get("title") or "",
        "speakers": list(fm.get("speakers") or []),
        "path": str(path),
    }


def build_output(frontmatter: dict, body: str) -> str:
    """Build transcript text with YAML front matter."""
    fm_text = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True).strip()
    return f"---\n{fm_text}\n---\n{body}"


def frontmatter_from_recording(rec: Recording) -> dict:
    """Create initial front matter dict from a Recording."""
    return {
        "recording_id": rec.short_id,
        "recorded": rec.date.astimezone().isoformat(),
        "duration": rec.duration.total_seconds(),
        "title": rec.title,
    }


def ensure_frontmatter(transcript_path: Path, rec: Recording) -> tuple[dict, str]:
    """Read transcript, add front matter if missing, write back. Returns (frontmatter, body)."""
    text = transcript_path.read_text()
    fm, body = parse_frontmatter(text)
    if fm is not None:
        return fm, body
    fm = frontmatter_from_recording(rec)
    transcript_path.write_text(build_output(fm, body))
    return fm, body
