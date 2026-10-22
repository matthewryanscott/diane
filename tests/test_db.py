"""Tests for recording discovery from filename timestamps."""

import hashlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from diane import db


@pytest.fixture
def recording_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "RECORDINGS_DIR", tmp_path)
    monkeypatch.setattr(db, "_birthtime", lambda path: None)
    probe = Mock(return_value=12.5)
    monkeypatch.setattr(db, "_ffprobe_duration", probe)
    return tmp_path, probe


def test_get_recordings_filename_dates_titles_ids_and_sorting(recording_directory):
    directory, probe = recording_directory
    filenames = [
        ("App Recording 20250415 0910.mp3", "App Recording", datetime(2025, 4, 15, 9, 10)),
        ("20260925 0905 Old Recording.m4a", "Old Recording", datetime(2026, 9, 25, 9, 5)),
        ("Voice Chat 20261002 0845.mp3", "Voice Chat", datetime(2026, 10, 2, 8, 45)),
    ]
    for filename, _, _ in filenames:
        (directory / filename).touch()

    recordings = db.get_recordings()

    assert [recording.path for recording in recordings] == [
        filename for filename, _, _ in reversed(filenames)
    ]
    for recording, (filename, title, date) in zip(recordings, reversed(filenames), strict=True):
        stem = (directory / filename).stem
        assert recording.unique_id == hashlib.sha256(stem.encode()).hexdigest()
        assert recording.short_id == recording.unique_id[:8]
        assert recording.title == title
        assert recording.date == date.astimezone()
        assert recording.duration == timedelta(seconds=12.5)
    assert probe.call_count == len(filenames)
    assert {call.args[0] for call in probe.call_args_list} == {
        directory / filename for filename, _, _ in filenames
    }


def test_get_recordings_date_first_takes_precedence(recording_directory):
    directory, _ = recording_directory
    filename = "20260925 0905 Voice Chat 20261002 0845.mp3"
    (directory / filename).touch()

    [recording] = db.get_recordings()

    assert recording.title == "Voice Chat 20261002 0845"
    assert recording.date == datetime(2026, 9, 25, 9, 5).astimezone()
    assert recording.unique_id == hashlib.sha256((directory / filename).stem.encode()).hexdigest()


def test_get_recordings_excludes_unmatched_and_undated_files(recording_directory):
    directory, probe = recording_directory
    for filename in [
        "Voice Chat.mp3",
        "20261002 Voice Chat.mp3",
        "Voice Chat 20261002.mp3",
        "Voice Chat 20261002 0845 2.mp3",
        ".Voice Chat 20261002 0845.mp3",
    ]:
        (directory / filename).touch()
    (directory / "App Recording 20250415 0910.mp3").mkdir()

    assert db.get_recordings() == []
    probe.assert_not_called()


@pytest.mark.parametrize(
    "filename",
    ["Voice Chat 20261002 0845.mp3", "20260925 0905 Old Recording.m4a"],
)
def test_get_recordings_prefers_birthtime(recording_directory, monkeypatch, filename):
    directory, _ = recording_directory
    path = directory / filename
    path.touch()
    birthtime = datetime(2024, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
    birthtime_reader = Mock(return_value=birthtime)
    monkeypatch.setattr(db, "_birthtime", birthtime_reader)

    [recording] = db.get_recordings()

    assert recording.date == birthtime
    birthtime_reader.assert_called_once_with(path)


@pytest.fixture
def association_directory(tmp_path, monkeypatch):
    recordings = tmp_path / "audio"
    recordings.mkdir()
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    monkeypatch.setattr(db, "RECORDINGS_DIR", recordings)
    monkeypatch.setattr(
        db.subprocess, "run", Mock(return_value=Mock(stdout='{"format": {"duration": "12.5"}}'))
    )
    return recordings, transcripts


@pytest.mark.parametrize(
    ("filename", "suffix_or_stem", "delimited_only"),
    [
        ("20261002T084500-a1b2c3d4.txt", "a1b2c3d4", "a1b2c3d4"),
        ("legacy.txt", "legacy", ""),
        ("legacy-more-parts.txt", "more-parts", "more-parts"),
        ("stamp-ABCD1234.txt", "ABCD1234", "ABCD1234"),
        ("stamp-.txt", "", ""),
    ],
)
def test_transcript_id_preserves_named_legacy_policies(
    tmp_path, filename, suffix_or_stem, delimited_only
):
    path = tmp_path / filename

    assert db.transcript_id(path) == suffix_or_stem
    assert db.transcript_id(path, policy="suffix-or-stem") == suffix_or_stem
    assert db.transcript_id(path, policy="delimited-only") == delimited_only


def test_canonical_transcript_path_uses_local_seconds_and_unknown_fallback(tmp_path):
    recording = db.Recording(
        unique_id="a1b2c3d4" + "0" * 56,
        date=datetime(2026, 10, 2, 8, 45, 37).astimezone(),
        duration=timedelta(seconds=12.5),
        title="Meeting",
        path="Meeting 20261002 0845.m4a",
    )

    assert db.canonical_transcript_path(recording, tmp_path) == (
        tmp_path / "20261002T084537-a1b2c3d4.txt"
    )
    recording.date = None
    assert db.canonical_transcript_path(recording, tmp_path) == tmp_path / "unknown-a1b2c3d4.txt"


def test_listing_status_lookup_and_pending_share_filename_ids(association_directory):
    audio, transcripts = association_directory
    source = audio / "Meeting 20261002 0845.m4a"
    source.write_bytes(b"synthetic audio")
    before = source.stat()
    [recording] = db.get_recordings()

    assert db.list_transcripts(transcripts) == []
    assert db.transcribed_ids(transcripts) == set()
    assert db.pending_recordings(transcripts) == [recording]

    path = db.canonical_transcript_path(recording, transcripts)
    path.write_text("---\nrecording_id: a-different-id\n---\nMeeting transcript\n")
    [entry] = db.list_transcripts(transcripts, include_recordings=True)

    assert entry == db.Transcript(path, recording.short_id, recording)
    assert db.list_transcripts(transcripts) == [db.Transcript(path, recording.short_id)]
    assert db.transcribed_ids(transcripts) == {recording.short_id}
    assert db.pending_recordings(transcripts) == []
    assert db.find_recording(recording.short_id[:4].upper()) == recording
    assert db.find_transcript(recording.short_id[:4].upper(), transcripts) == (
        db.Transcript(path, recording.short_id)
    )
    assert db.find_recording_transcript(recording.short_id[:4].upper(), transcripts) == entry
    assert source.read_bytes() == b"synthetic audio"
    after = source.stat()
    assert (after.st_mtime_ns, after.st_ctime_ns, after.st_size, after.st_mode) == (
        before.st_mtime_ns, before.st_ctime_ns, before.st_size, before.st_mode
    )


def test_noncanonical_transcript_skips_bulk_but_does_not_resolve_canonical(
    association_directory,
):
    audio, transcripts = association_directory
    (audio / "20261002 0845 Meeting.m4a").touch()
    [recording] = db.get_recordings()
    alternate = transcripts / f"old-timestamp-{recording.short_id}.txt"
    alternate.touch()

    # Only the first hyphen delimits an ID; a later matching suffix is not a join.
    assert db.transcribed_ids(transcripts) == {f"timestamp-{recording.short_id}"}
    assert db.pending_recordings(transcripts) == [recording]
    alternate.unlink()
    alternate = transcripts / f"oldtimestamp-{recording.short_id}.txt"
    alternate.touch()

    assert not db.canonical_transcript_path(recording, transcripts).exists()
    assert db.transcribed_ids(transcripts) == {recording.short_id}
    assert db.pending_recordings(transcripts) == []
    assert db.find_transcript(recording.short_id, transcripts).path == alternate
    with pytest.raises(FileNotFoundError) as error:
        db.find_recording_transcript(recording.short_id.upper(), transcripts)
    assert str(error.value) == (
        f"No transcript for {recording.short_id.upper()}. "
        f"Run 'diane transcribe {recording.short_id.upper()}' first."
    )


def test_orphan_listing_and_path_lookup_never_discover_recordings(tmp_path, monkeypatch):
    unavailable_audio = tmp_path / "not-a-directory"
    unavailable_audio.touch()
    monkeypatch.setattr(db, "RECORDINGS_DIR", unavailable_audio)
    probe = Mock(side_effect=AssertionError("transcript-only operations must not probe audio"))
    monkeypatch.setattr(db.subprocess, "run", probe)
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    orphan = transcripts / "20261002T084500-a1b2c3d4.txt"
    orphan.write_text("orphan transcript")

    assert db.list_transcripts(transcripts) == [db.Transcript(orphan, "a1b2c3d4")]
    assert db.list_transcripts(transcripts, include_recordings=False) == [
        db.Transcript(orphan, "a1b2c3d4")
    ]
    assert db.find_transcript("A1B2", transcripts) == db.Transcript(orphan, "a1b2c3d4")
    assert db.transcribed_ids(transcripts) == {"a1b2c3d4"}
    probe.assert_not_called()


@pytest.mark.parametrize("directory_kind", ["missing", "empty", "non-transcript"])
def test_empty_transcript_enumeration_never_discovers_recordings(
    tmp_path, monkeypatch, directory_kind
):
    unavailable_audio = tmp_path / "not-a-directory"
    unavailable_audio.touch()
    monkeypatch.setattr(db, "RECORDINGS_DIR", unavailable_audio)
    transcripts = tmp_path / "transcripts"
    if directory_kind != "missing":
        transcripts.mkdir()
    if directory_kind == "non-transcript":
        (transcripts / "ignored.json").touch()

    assert db.list_transcripts(transcripts, include_recordings=True) == []


def test_nonempty_enrichment_propagates_recording_discovery_errors(tmp_path, monkeypatch):
    unavailable_audio = tmp_path / "not-a-directory"
    unavailable_audio.touch()
    monkeypatch.setattr(db, "RECORDINGS_DIR", unavailable_audio)
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    (transcripts / "stamp-deadbeef.txt").touch()

    with pytest.raises(NotADirectoryError):
        db.list_transcripts(transcripts, include_recordings=True)


def test_no_hyphen_policy_controls_enrichment_but_not_bulk_status(association_directory):
    audio, transcripts = association_directory
    (audio / "20261002 0845 Meeting.m4a").touch()
    [recording] = db.get_recordings()
    path = transcripts / f"{recording.short_id}.txt"
    path.touch()

    assert db.list_transcripts(transcripts, include_recordings=True) == [
        db.Transcript(path, recording.short_id, recording)
    ]
    assert db.list_transcripts(
        transcripts, id_policy="delimited-only", include_recordings=True
    ) == [db.Transcript(path, "")]
    assert db.find_transcript(recording.short_id.upper(), transcripts).path == path
    assert db.transcribed_ids(transcripts) == {recording.short_id}
    assert db.pending_recordings(transcripts) == []
    with pytest.raises(FileNotFoundError):
        db.find_recording_transcript(recording.short_id, transcripts)


def test_exact_filename_id_membership_does_not_casefold_or_match_prefixes(
    association_directory,
):
    audio, transcripts = association_directory
    (audio / "20261002 0845 Meeting.m4a").touch()
    [recording] = db.get_recordings()
    paths = [
        transcripts / f"stamp-{recording.short_id[:4]}.txt",
        transcripts / f"stamp-{recording.short_id}extra.txt",
    ]
    for path in paths:
        path.touch()
    # A noncanonical association ID has no normalized/shortened fallback.
    entries = db.list_transcripts(transcripts, include_recordings=True)

    assert all(entry.recording is None for entry in entries)
    assert db.pending_recordings(transcripts) == [recording]
    for path in paths:
        path.unlink()
    uppercase = transcripts / f"stamp-{recording.short_id.upper()}.txt"
    uppercase.touch()
    [entry] = db.list_transcripts(transcripts, include_recordings=True)
    assert entry.recording is None
    assert db.pending_recordings(transcripts) == [recording]
    assert db.find_transcript(recording.short_id, transcripts).path == uppercase


def test_transcript_prefix_missing_ambiguous_and_empty_outcomes(tmp_path):
    transcripts = tmp_path / "missing"

    assert db.list_transcripts(transcripts) == []
    assert db.transcribed_ids(transcripts) == set()
    with pytest.raises(FileNotFoundError, match=r"^No transcript for ab\.$"):
        db.find_transcript("AB", transcripts)
    transcripts.mkdir()
    first = transcripts / "stamp-AbCd1234.txt"
    first.touch()
    assert db.find_transcript("", transcripts).path == first
    assert db.find_transcript("ABC", transcripts).path == first
    with pytest.raises(FileNotFoundError, match=r"^No transcript for ffff\.$"):
        db.find_transcript("FFFF", transcripts)
    (transcripts / "another-abce5678.txt").touch()
    for prefix in ("ABC", ""):
        with pytest.raises(ValueError) as error:
            db.find_transcript(prefix, transcripts)
        assert str(error.value) == (
            f"Ambiguous ID '{prefix.lower()}' matches 2 transcripts"
        )


def test_duplicate_transcript_ids_remain_ambiguous_for_path_lookup(association_directory):
    audio, transcripts = association_directory
    (audio / "20261002 0845 Meeting.m4a").touch()
    [recording] = db.get_recordings()
    canonical = db.canonical_transcript_path(recording, transcripts)
    canonical.touch()
    (transcripts / f"renamed-{recording.short_id}.txt").touch()

    with pytest.raises(ValueError, match="matches 2 transcripts"):
        db.find_transcript(recording.short_id, transcripts)
    assert db.find_recording_transcript(recording.short_id, transcripts).path == canonical
    assert db.transcribed_ids(transcripts) == {recording.short_id}


def test_metadata_dependent_lookup_requires_recording_even_for_orphan(association_directory):
    _, transcripts = association_directory
    orphan = transcripts / "stamp-a1b2c3d4.txt"
    orphan.touch()

    assert db.find_transcript("a1b2", transcripts).path == orphan
    with pytest.raises(ValueError, match="^No recording found matching 'a1b2'$"):
        db.find_recording_transcript("A1B2", transcripts)


def test_recording_prefix_missing_ambiguous_and_empty_outcomes(association_directory):
    audio, transcripts = association_directory

    with pytest.raises(ValueError, match="^No recording found matching 'ffff'$"):
        db.find_recording("FFFF")
    (audio / "20261002 0845 Meeting.m4a").touch()
    [recording] = db.get_recordings()
    assert db.find_recording("") == recording
    assert db.find_recording(recording.short_id.upper()) == recording
    (audio / "Second Meeting 20261003 0930.mp3").touch()
    with pytest.raises(ValueError, match="^Ambiguous ID '' matches 2 recordings$"):
        db.find_recording("")
    with pytest.raises(ValueError, match="^Ambiguous ID '' matches 2 recordings$"):
        db.find_recording_transcript("", transcripts)


def test_pending_recordings_excludes_audio_removed_during_discovery(
    association_directory, monkeypatch
):
    audio, transcripts = association_directory
    missing = audio / "20261002 0845 Removed Meeting.m4a"
    available = audio / "20261003 0930 Available Meeting.m4a"
    missing.touch()
    available.touch()

    def probe(command, **kwargs):
        if command[-1] == str(missing):
            missing.unlink()
        return Mock(stdout='{"format": {"duration": "12.5"}}')

    monkeypatch.setattr(db.subprocess, "run", probe)

    pending = db.pending_recordings(transcripts)

    assert [recording.path for recording in pending] == [available.name]
    assert not missing.exists()
    assert available.exists()


def test_transcript_enumeration_is_direct_and_keeps_legacy_glob_entries(tmp_path):
    paths = [
        tmp_path / "z-orphan.txt",
        tmp_path / "a-orphan.txt",
        tmp_path / ".hidden.txt",
        tmp_path / ".txt",
    ]
    for path in paths:
        path.touch()
    (tmp_path / "ignored.TXT").touch()
    (tmp_path / "ignored.json").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "stamp-nested.txt").touch()
    directory_entry = tmp_path / "directory-id.txt"
    directory_entry.mkdir()

    entries = db.list_transcripts(tmp_path)

    assert {entry.path for entry in entries} == {*paths, directory_entry}
    assert all(entry.recording is None for entry in entries)


def test_directory_errors_other_than_missing_root_propagate(tmp_path, monkeypatch):
    not_directory = tmp_path / "file"
    not_directory.touch()
    monkeypatch.setattr(db, "RECORDINGS_DIR", not_directory)

    with pytest.raises(NotADirectoryError):
        db.get_recordings()
    with pytest.raises(NotADirectoryError):
        db.list_transcripts(not_directory)
    with pytest.raises(NotADirectoryError):
        db.find_transcript("abcd", not_directory)
    with pytest.raises(NotADirectoryError):
        db.transcribed_ids(not_directory)


def test_missing_recordings_root_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "RECORDINGS_DIR", tmp_path / "missing-audio")

    assert db.get_recordings() == []
    assert db.pending_recordings(tmp_path / "missing-transcripts") == []


def test_discovery_follows_file_symlinks_without_recursing_or_filtering_extensions(
    association_directory,
):
    audio, _ = association_directory
    target = audio.parent / "source"
    target.write_bytes(b"source audio")
    linked = audio / "Linked Meeting 20261002 0845.unusual"
    linked.symlink_to(target)
    direct = audio / "20261003 0930 Café Meeting.binary"
    direct.touch()
    nested = audio / "nested"
    nested.mkdir()
    (nested / "20261004 1000 Nested Meeting.m4a").touch()
    (audio / "Broken Meeting 20261002 0845.m4a").symlink_to(audio / "absent")

    recordings = db.get_recordings()

    assert {recording.path for recording in recordings} == {linked.name, direct.name}
    assert {recording.title for recording in recordings} == {"Linked Meeting", "Café Meeting"}
    assert target.read_bytes() == b"source audio"


def test_recording_ids_ignore_extension(recording_directory):
    directory, _ = recording_directory
    for filename in ["20261002 0845 Meeting.m4a", "20261002 0845 Meeting.arbitrary"]:
        (directory / filename).touch()

    first, second = db.get_recordings()

    assert first.unique_id == second.unique_id
    assert first.short_id == second.short_id
    with pytest.raises(ValueError, match="matches 2 recordings"):
        db.find_recording(first.short_id)


@pytest.mark.parametrize("failure", [OSError("ffprobe unavailable"), ValueError("bad duration")])
def test_duration_probe_failure_keeps_recording_with_zero_duration(
    association_directory, monkeypatch, failure
):
    audio, _ = association_directory
    (audio / "20261002 0845 Meeting.m4a").touch()
    monkeypatch.setattr(db.subprocess, "run", Mock(side_effect=failure))

    [recording] = db.get_recordings()

    assert recording.duration == timedelta(0)


@pytest.mark.parametrize(
    "filename",
    ["20261340 2561 Meeting.m4a", "Meeting 20261340 2561.m4a"],
)
def test_structural_filename_match_does_not_silently_validate_invalid_dates(
    recording_directory, filename
):
    directory, probe = recording_directory
    (directory / filename).touch()

    with pytest.raises(ValueError):
        db.get_recordings()
    probe.assert_not_called()


def test_usable_birthtime_bypasses_invalid_numeric_filename_date(
    recording_directory, monkeypatch
):
    directory, _ = recording_directory
    (directory / "20261340 2561 Meeting.m4a").touch()
    birthtime = datetime(2024, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
    monkeypatch.setattr(db, "_birthtime", lambda path: birthtime)

    [recording] = db.get_recordings()

    assert recording.date == birthtime
    assert recording.title == "Meeting"


@pytest.mark.parametrize("failure", ["missing-attribute", "os-error"])
def test_birthtime_stat_failures_fall_back_to_local_filename_date(
    association_directory, monkeypatch, failure
):
    audio, _ = association_directory
    source = audio / "20261002 0845 Meeting.m4a"
    source.touch()
    real_stat = db.os.stat
    source_calls = 0

    def stat(path, *args, **kwargs):
        nonlocal source_calls
        result = real_stat(path, *args, **kwargs)
        if path == source:
            source_calls += 1
            if source_calls == 2:
                if failure == "os-error":
                    raise OSError("birthtime unavailable")
                return SimpleNamespace(st_mode=result.st_mode)
        return result

    monkeypatch.setattr(db.os, "stat", stat)

    [recording] = db.get_recordings()

    assert recording.date == datetime(2026, 10, 2, 8, 45).astimezone()
    assert recording.date.tzinfo is not None


def test_pending_selection_preserves_newest_first_recording_order(recording_directory, tmp_path):
    audio, _ = recording_directory
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    for filename in [
        "20261001 0800 Old Meeting.m4a",
        "20261003 0800 New Meeting.m4a",
        "20261002 0800 Already Transcribed.m4a",
    ]:
        (audio / filename).touch()
    newest, transcribed, oldest = db.get_recordings()
    db.canonical_transcript_path(transcribed, transcripts).touch()

    assert db.pending_recordings(transcripts) == [newest, oldest]
