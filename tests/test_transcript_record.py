"""Tests for the machine-readable transcript record behind `transcript list --json`."""

from diane.frontmatter import transcript_record


def _write(path, text):
    path.write_text(text)
    return path


def test_record_from_frontmatter(tmp_path):
    path = _write(
        tmp_path / "20260101T120000-1234abcd.txt",
        "---\n"
        "duration: 12.5\n"
        "recorded: '2026-01-01T12:00:00-05:00'\n"
        "recording_id: 1234abcd\n"
        "speakers:\n- Practice A\n- Practice B\n"
        "title: meeting\n"
        "---\n"
        "[00:01] Practice A: Hi.\n",
    )
    assert transcript_record(path) == {
        "id": "1234abcd",
        "recorded": "2026-01-01T12:00:00-05:00",
        "duration_seconds": 12.5,
        "title": "meeting",
        "speakers": ["Practice A", "Practice B"],
        "path": str(path),
    }


def test_record_without_frontmatter_falls_back_to_filename(tmp_path):
    path = _write(tmp_path / "20260401T090000-abcd1234.txt", "[00:01] Practice A: no front matter\n")
    assert transcript_record(path) == {
        "id": "abcd1234",
        "recorded": None,
        "duration_seconds": None,
        "title": "",
        "speakers": [],
        "path": str(path),
    }
