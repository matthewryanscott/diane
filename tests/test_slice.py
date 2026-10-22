"""Tests for the top-level slice command."""

from datetime import datetime

from diane.timestamps import extract_timestamp, timestamp_line_content


def test_extract_timestamp_with_absolute():
    line = "[2026-04-02T17:40:58] Practice A: Hello world"
    result = extract_timestamp(line)
    assert result == datetime(2026, 4, 2, 17, 40, 58)


def test_extract_timestamp_with_no_timestamp():
    assert extract_timestamp("Just some text") is None
    assert extract_timestamp("") is None


def test_extract_timestamp_with_relative():
    assert extract_timestamp("[05:30] Practice A: Hello") is None


def test_timestamp_line_content_with_content():
    assert timestamp_line_content("[2026-04-03T08:00:38] Engine noise") == "Engine noise"


def test_timestamp_line_content_empty():
    assert timestamp_line_content("[2026-04-03T08:00:38] ") == ""


def test_timestamp_line_content_no_space():
    assert timestamp_line_content("[2026-04-03T08:00:38]") == ""


def test_timestamp_line_content_whitespace_only():
    assert timestamp_line_content("[2026-04-03T08:00:38]   \t") == ""


def test_timestamp_line_content_no_timestamp():
    assert timestamp_line_content("plain text") is None


def test_slice_source_tagging(tmp_path, monkeypatch):
    """Slice output should include [source] tag from parent directory name."""
    import diane.db

    monkeypatch.setattr(diane.db, "PROJECT_ROOT", tmp_path)

    # Also patch PROJECT_ROOT in cli module since it's imported at module level
    import diane.cli

    monkeypatch.setattr(diane.cli, "PROJECT_ROOT", tmp_path)

    # Create two source dirs with absolute-timestamped files
    for subdir, content in [
        ("transcripts", "[2026-04-02T17:00:00] Practice A: Hello from transcripts"),
        ("background", "[2026-04-02T17:00:05] Engine noise detected"),
    ]:
        d = tmp_path / subdir
        d.mkdir()
        (d / "20260402T170000-abcd1234.txt").write_text(
            f"---\ntimestamps: absolute\n---\n{content}\n"
        )

    from diane.cli import slice as slice_cmd
    from io import StringIO
    import sys

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    slice_cmd(start="2026-04-02T16:59:00", end="2026-04-02T17:01:00")

    output = captured.getvalue()
    assert "[transcripts]" in output
    assert "[background]" in output
    assert output.index("[transcripts]") < output.index("[background]")


def test_absolute_conversion_output_can_be_sliced(tmp_path, monkeypatch, capsys):
    """Meeting timestamp conversion remains usable without any calendar setup."""
    import wave
    from datetime import timedelta

    import diane.cli
    import diane.db
    from diane.frontmatter import parse_frontmatter

    recordings = tmp_path / "audio"
    recordings.mkdir()
    with wave.open(str(recordings / "20260402 1700 Meeting.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * 80)
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    monkeypatch.setattr(diane.db, "RECORDINGS_DIR", recordings)
    monkeypatch.setattr(diane.cli, "RECORDINGS_DIR", recordings)
    monkeypatch.setattr(diane.cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(diane.cli, "TRANSCRIPTS_DIR", transcripts)
    rec, = diane.db.get_recordings()
    transcript = transcripts / (
        f"{rec.date.astimezone():%Y%m%dT%H%M%S}-{rec.short_id}.txt"
    )
    transcript.write_text("[00:02] Practice A: Keep this turn\n[01:00] Practice A: Outside range\n")

    diane.cli.app(
        ["transcript", "absolute", rec.short_id], result_action="return_value",
    )

    fm, body = parse_frontmatter(transcript.read_text())
    assert fm["timestamps"] == "absolute"
    assert fm["recorded"] == rec.date.isoformat()
    first_timestamp = extract_timestamp(body.splitlines()[0])
    assert first_timestamp - rec.date.replace(tzinfo=None, microsecond=0) == timedelta(seconds=2)
    capsys.readouterr()

    # Relative sources must not accidentally enter an absolute-time search.
    (transcripts / "relative.txt").write_text("[00:02] Practice A: Relative source\n")
    diane.cli.app([
        "slice",
        "--start", rec.date.isoformat(),
        "--end", (rec.date + timedelta(seconds=10)).isoformat(),
    ], result_action="return_value")

    output = capsys.readouterr().out
    assert "[transcripts] Practice A: Keep this turn" in output
    assert "Outside range" not in output
    assert "Relative source" not in output
