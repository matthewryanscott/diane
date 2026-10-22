"""Behavioral coverage for the public command surface."""

import json
import wave
from types import SimpleNamespace

import pytest

from diane.cli import app


@pytest.mark.parametrize("command", [[], ["list"], ["events"], ["add"], ["remove"], ["gaps"]])
def test_calendar_commands_are_rejected(command, capsys):
    with pytest.raises(SystemExit) as exc:
        app(["calendar", *command])
    assert exc.value.code != 0
    output = capsys.readouterr()
    assert "calendar" in output.err + output.out


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (["--help"], ["transcribe", "transcript", "slice", "transcript list --json"]),
        (["transcript", "--help"], ["absolute", "list", "path"]),
        (["transcript", "absolute", "--help"], ["recording", "all"]),
        (["slice", "--help"], ["--start", "--end"]),
    ],
)
def test_help_exposes_retained_commands_only(command, expected, capsys):
    try:
        app(command)
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    output = captured.out + captured.err
    for text in expected:
        assert text in output
    assert "calendar" not in output.lower()
    assert "--day-start" not in output
    assert "--day-end" not in output


@pytest.fixture
def discovery_roots(tmp_path, monkeypatch):
    """Use real audio and discovery; redirect only the configured filesystem roots."""
    import diane.cli
    import diane.db

    recordings = tmp_path / "audio"
    recordings.mkdir()
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    for name in ["20260402 1700 Pending meeting.wav", "Finished meeting 20260401 0900.wav"]:
        with wave.open(str(recordings / name), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(b"\x00\x00" * 64000)
    monkeypatch.setattr(diane.db, "RECORDINGS_DIR", recordings)
    monkeypatch.setattr(diane.cli, "RECORDINGS_DIR", recordings)
    monkeypatch.setattr(diane.cli, "TRANSCRIPTS_DIR", transcripts)
    monkeypatch.setenv("COLUMNS", "240")
    monkeypatch.setenv("NO_COLOR", "1")

    def source_state():
        state = {}
        for path in recordings.iterdir():
            content = path.read_bytes()
            stat = path.stat()
            state[path.name] = (
                content, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                stat.st_mode, getattr(stat, "st_birthtime", None),
            )
        return state

    original_sources = source_state()
    discovered = diane.db.get_recordings()
    recs = {rec.title: rec for rec in discovered}
    assert set(recs) == {"Pending meeting", "Finished meeting"}
    yield SimpleNamespace(
        recordings=recordings,
        transcripts=transcripts,
        discovered=discovered,
        pending=recs["Pending meeting"],
        finished=recs["Finished meeting"],
    )
    assert source_state() == original_sources


def test_transcribe_dry_run_selects_real_recordings(discovery_roots, capsys):
    from diane.db import canonical_transcript_path

    roots = discovery_roots
    pending, finished = roots.pending, roots.finished

    app(["transcribe", "all", "--dry-run"], result_action="return_value")
    output = capsys.readouterr()
    assert "Pending meeting" in output.err + output.out
    assert "Finished meeting" in output.err + output.out
    assert list(roots.transcripts.iterdir()) == []

    finished_path = canonical_transcript_path(finished, roots.transcripts)
    finished_path.write_text("[00:00] Practice A: Already done\n")

    app(["transcribe", pending.short_id.upper(), "--dry-run"], result_action="return_value")
    output = capsys.readouterr()
    assert "Would transcribe" in output.err + output.out
    assert "Pending meeting" in output.err + output.out

    app(["transcribe", "all", "--dry-run"], result_action="return_value")
    output = capsys.readouterr()
    assert "Pending meeting" in output.err + output.out
    assert "Finished meeting" not in output.err + output.out

    app(["transcribe", finished.short_id, "--dry-run"], result_action="return_value")
    output = capsys.readouterr()
    assert "Transcript already exists" in output.err + output.out

    with pytest.raises(SystemExit) as exc:
        app(["transcribe", "missing-recording", "--dry-run"])
    assert exc.value.code != 0
    assert list(roots.transcripts.iterdir()) == [finished_path]
    assert finished_path.read_text() == "[00:00] Practice A: Already done\n"


@pytest.mark.parametrize("filename_kind", ["canonical", "noncanonical", "no-hyphen"])
def test_recording_status_and_bulk_skip_use_filename_id(discovery_roots, capsys, filename_kind):
    from diane.db import canonical_transcript_path

    roots = discovery_roots
    app(["recording", "list", "--limit", "0"], result_action="return_value")
    before = capsys.readouterr().out
    assert before.index(roots.discovered[0].short_id) < before.index(roots.discovered[1].short_id)
    assert all(not line.rstrip().endswith("Y") for line in before.splitlines())

    if filename_kind == "canonical":
        path = canonical_transcript_path(roots.finished, roots.transcripts)
    elif filename_kind == "noncanonical":
        path = roots.transcripts / f"19990101T000000-{roots.finished.short_id}.txt"
    else:
        path = roots.transcripts / f"{roots.finished.short_id}.txt"
    # JSON's displayed ID must not change filename-based association or skip rules.
    text = f"---\nrecording_id: {roots.pending.short_id}\n---\n[00:00] Practice A: Done\n"
    path.write_text(text)

    app(["recording", "list", "--limit", "0"], result_action="return_value")
    listing = capsys.readouterr().out
    finished_row = next(line for line in listing.splitlines() if roots.finished.short_id in line)
    pending_row = next(line for line in listing.splitlines() if roots.pending.short_id in line)
    assert finished_row.rstrip().endswith("Y")
    assert not pending_row.rstrip().endswith("Y")

    app(["transcribe", "all", "--dry-run"], result_action="return_value")
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "Pending meeting" in output
    assert "Finished meeting" not in output

    app(["transcribe", roots.finished.short_id, "--dry-run"], result_action="return_value")
    captured = capsys.readouterr()
    output = captured.out + captured.err
    if filename_kind == "canonical":
        assert "Transcript already exists" in output
    else:
        assert "Would transcribe" in output
    assert path.read_text() == text
    assert list(roots.transcripts.iterdir()) == [path]


def test_transcript_list_keeps_orphans_and_legacy_ids_in_reverse_path_order(discovery_roots, capsys):
    roots = discovery_roots
    matched = roots.transcripts / f"20000101T000000-{roots.finished.short_id}.txt"
    orphan = roots.transcripts / "20010101T000000-deadbeef.txt"
    legacy = roots.transcripts / f"{roots.pending.short_id}.txt"
    matched.write_text(
        f"---\nrecording_id: {roots.pending.short_id}\ntitle: Display title\n---\n"
        "[00:00] Practice A: Done\n"
    )
    orphan.write_text("[00:00] Practice A: No recording\n")
    legacy.write_text("[00:00] Practice A: Legacy filename\n")
    expected_paths = sorted([matched, orphan, legacy], reverse=True)

    app(["transcript", "list"], result_action="return_value")
    listing = capsys.readouterr().out
    assert [listing.index(path.name) for path in expected_paths] == sorted(
        listing.index(path.name) for path in expected_paths
    )
    assert "Finished meeting" in listing
    assert "Pending meeting" not in listing  # No-hyphen table association is empty.
    assert "deadbeef" in listing
    assert "3 transcript(s)" in listing
    legacy_row = next(line for line in listing.splitlines() if legacy.name in line)
    assert legacy_row.replace("│", "").strip() == legacy.name

    app(["transcript", "list", "--json"], result_action="return_value")
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["path"] for record in records] == [str(path) for path in expected_paths]
    by_path = {record["path"]: record for record in records}
    assert by_path[str(matched)]["id"] == roots.pending.short_id
    assert by_path[str(matched)]["title"] == "Display title"
    assert by_path[str(orphan)]["id"] == "deadbeef"
    assert by_path[str(legacy)]["id"] == roots.pending.short_id


def test_empty_transcript_table_does_not_discover_recordings(tmp_path, monkeypatch, capsys):
    import diane.cli
    import diane.db

    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    unavailable_audio = tmp_path / "not-a-directory"
    unavailable_audio.touch()
    monkeypatch.setattr(diane.cli, "TRANSCRIPTS_DIR", transcripts)
    monkeypatch.setattr(diane.cli, "RECORDINGS_DIR", unavailable_audio)
    monkeypatch.setattr(diane.db, "RECORDINGS_DIR", unavailable_audio)

    app(["transcript", "list"], result_action="return_value")

    captured = capsys.readouterr()
    assert "No transcripts yet." in captured.out + captured.err


def test_orphan_table_listing_works_without_recordings(discovery_roots, monkeypatch, capsys):
    import diane.db

    roots = discovery_roots
    orphan = roots.transcripts / "20000101T000000-deadbeef.txt"
    orphan.write_text("[00:00] Practice A: No source\n")
    monkeypatch.setattr(diane.db, "RECORDINGS_DIR", roots.recordings / "absent")
    app(["transcript", "list"], result_action="return_value")
    listing = capsys.readouterr().out
    assert orphan.name in listing
    assert "1 transcript(s)" in listing


def test_transcript_path_and_json_do_not_discover_recordings(discovery_roots, monkeypatch, capsys):
    import diane.db

    roots = discovery_roots
    orphan = roots.transcripts / "20000101T000000-deadbeef.txt"
    legacy = roots.transcripts / "cafebabe.txt"
    orphan.write_text("[00:00] Practice A: No source\n")
    legacy.write_text("[00:00] Practice A: Legacy\n")
    # An invalid directory would fail if either command tried recording discovery.
    monkeypatch.setattr(diane.db, "RECORDINGS_DIR", roots.recordings / roots.pending.path)
    for prefix, path in [("DEAD", orphan), ("CAFE", legacy)]:
        app(["transcript", "path", prefix], result_action="return_value")
        assert capsys.readouterr().out.strip() == str(path.resolve())

    app(["transcript", "list", "--json"], result_action="return_value")
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert {record["id"] for record in records} == {"deadbeef", "cafebabe"}

    duplicate = roots.transcripts / "20010101T000000-deadbeef.txt"
    duplicate.write_text("Another transcript\n")
    for prefix, expected in [("dead", "Ambiguous"), ("missing-id", "No transcript")]:
        with pytest.raises(SystemExit) as exc:
            app(["transcript", "path", prefix])
        assert exc.value.code != 0
        captured = capsys.readouterr()
        assert expected in captured.out + captured.err


def test_metadata_dependent_commands_require_canonical_path(discovery_roots, capsys):
    roots = discovery_roots
    noncanonical = roots.transcripts / f"19990101T000000-{roots.finished.short_id}.txt"
    text = "[00:00] SPEAKER_00: Noncanonical transcript\n"
    noncanonical.write_text(text)

    app(["transcript", "path", roots.finished.short_id], result_action="return_value")
    assert capsys.readouterr().out.strip() == str(noncanonical.resolve())

    app(["speakers", roots.finished.short_id], result_action="return_value")
    captured = capsys.readouterr()
    assert "Warning:" in captured.out + captured.err
    assert "No transcripts to process" in captured.out + captured.err

    with pytest.raises(SystemExit) as exc:
        app(["transcript", "absolute", roots.finished.short_id])
    assert exc.value.code != 0
    assert noncanonical.read_text() == text


def test_absolute_all_uses_ascending_paths_and_skips_legacy_orphans(discovery_roots, capsys):
    from diane.frontmatter import parse_frontmatter

    roots = discovery_roots
    earlier_path = roots.transcripts / f"19990101T000000-{roots.pending.short_id}.txt"
    later_path = roots.transcripts / f"20000101T000000-{roots.finished.short_id}.txt"
    legacy = roots.transcripts / f"{roots.pending.short_id}.txt"
    orphan = roots.transcripts / "20010101T000000-deadbeef.txt"
    text = "[00:00] Practice A: Opening\n[00:01] Practice A: Next\n"
    for path in [later_path, legacy, orphan, earlier_path]:
        path.write_text(text)

    app(["transcript", "absolute", "all"], result_action="return_value")
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert output.index(earlier_path.name) < output.index(later_path.name)
    assert "2 converted" in output
    for path, recording in [(earlier_path, roots.pending), (later_path, roots.finished)]:
        frontmatter, body = parse_frontmatter(path.read_text())
        assert frontmatter["timestamps"] == "absolute"
        assert frontmatter["recording_id"] == recording.short_id
        assert recording.date.astimezone().strftime("[%Y-%m-%dT%H:%M:%S]") in body
    assert legacy.read_text() == text
    assert orphan.read_text() == text


def test_speakers_all_uses_reverse_paths_and_skips_legacy_orphans(
    discovery_roots, monkeypatch, capsys
):
    import diane.speakers
    from diane.frontmatter import parse_frontmatter

    roots = discovery_roots
    earlier_path = roots.transcripts / f"19990101T000000-{roots.pending.short_id}.txt"
    later_path = roots.transcripts / f"20000101T000000-{roots.finished.short_id}.txt"
    legacy = roots.transcripts / f"{roots.pending.short_id}.txt"
    orphan = roots.transcripts / "20010101T000000-deadbeef.txt"
    text = "[00:00] SPEAKER_00: Opening\n[00:01] SPEAKER_00: Next\n"
    for path in [earlier_path, legacy, orphan, later_path]:
        path.write_text(text)
    # Human/audio interaction only; association and transcript processing stay real.
    monkeypatch.setattr(diane.speakers, "prompt_with_snippets", lambda *args, **kwargs: "Practice A")

    app(["speakers", "all"], result_action="return_value")
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert output.index(f"Wrote {later_path.name}") < output.index(f"Wrote {earlier_path.name}")
    for path in [earlier_path, later_path]:
        frontmatter, body = parse_frontmatter(path.read_text())
        assert frontmatter["speakers"] == ["Practice A"]
        assert "SPEAKER_00" not in body
        assert "Practice A: Opening" in body
    assert legacy.read_text() == text
    assert orphan.read_text() == text


def test_audit_prefix_filter_keeps_reverse_paths_and_no_hyphen_association(
    discovery_roots, monkeypatch, capsys
):
    import diane.speakers
    from diane.voiceprints import Cluster, Sidecar, read_sidecar, write_sidecar

    roots = discovery_roots
    earlier_path = roots.transcripts / f"19990101T000000-{roots.finished.short_id}.txt"
    later_path = roots.transcripts / f"20000101T000000-{roots.finished.short_id}.txt"
    legacy = roots.transcripts / f"{roots.finished.short_id}.txt"
    unrelated = roots.transcripts / f"20010101T000000-{roots.pending.short_id}.txt"
    text = "[00:00] Practice A: Opening\n[00:01] Practice A: Next\n"
    for path in [earlier_path, legacy, unrelated, later_path]:
        path.write_text(text)
        write_sidecar(path, Sidecar(clusters=[
            Cluster("SPEAKER_00", [1.0, 0.0], 1.0, name="Practice A", source="auto", score=0.99)
        ]))
    monkeypatch.setattr(diane.speakers, "prompt_with_snippets", lambda *args, **kwargs: "")

    # A prefix may select several transcripts; audit is a filter, not unique lookup.
    app(["speakers", "audit", roots.finished.short_id.upper()], result_action="return_value")
    captured = capsys.readouterr()
    output = captured.out + captured.err
    expected_paths = sorted([earlier_path, later_path, legacy], reverse=True)
    assert [output.index(path.name) for path in expected_paths] == sorted(
        output.index(path.name) for path in expected_paths
    )
    assert unrelated.name not in output
    for path in expected_paths:
        assert read_sidecar(path).clusters[0].source == "confirmed"
        assert "Practice A: Opening" in path.read_text()
    assert read_sidecar(unrelated).clusters[0].source == "auto"
    assert unrelated.read_text() == text


def test_transcription_failure_has_no_success_artifacts_and_batch_continues(
    discovery_roots, monkeypatch, capsys
):
    import scribe

    import diane.transcribe
    from diane.db import canonical_transcript_path
    from diane.frontmatter import parse_frontmatter
    from diane.voiceprints import read_sidecar, sidecar_path

    roots = discovery_roots
    monkeypatch.setattr(diane.transcribe, "RECORDINGS_DIR", roots.recordings)
    calls = 0

    def recognize(_audio):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise scribe.ScribeError("controlled recognition failure")
        return [scribe.TranscriptionSegment("Synthetic successful turn.", 0.0, 1.0)]

    class Embedder:
        def _generate_subsegments(self, regions, _accurate):
            assert regions == [(0.0, 2.3456)]
            return [(0.0, 1.0), (1.0, 2.0)]

        def _extract_fbank_features(self, path, windows):
            with wave.open(str(path), "rb") as audio:
                assert audio.getframerate() == 16000
                assert audio.getnchannels() == 1
                assert audio.getsampwidth() == 2
                assert audio.getnframes() == 128000
            assert windows == [(0.0, 1.0), (1.0, 2.0)]
            return None, None, [0, 1], 2

        def _generate_embeddings(self, *_args):
            return [[3.0, 4.0], [0.0, 5.0]]

    monkeypatch.setattr(scribe, "run_transcription", recognize)
    monkeypatch.setattr(
        scribe, "run_nemotron_diarization",
        lambda _audio: [
            scribe.DiarizationSegment("SPEAKER_00", 0.0, 1.5),
            scribe.DiarizationSegment("SPEAKER_00", 1.0, 2.3456),
        ],
    )
    monkeypatch.setattr(scribe, "_create_embedder", Embedder)

    app(["transcribe", "all"], result_action="return_value")

    captured = capsys.readouterr()
    output = captured.out + captured.err
    failed, succeeded = roots.discovered
    failed_path = canonical_transcript_path(failed, roots.transcripts)
    succeeded_path = canonical_transcript_path(succeeded, roots.transcripts)
    assert calls == 2
    assert "controlled recognition failure" in output
    assert "1 recording(s) failed" in output
    assert not failed_path.exists()
    assert not sidecar_path(failed_path).exists()
    assert output.count("Wrote") == 1
    fm, body = parse_frontmatter(succeeded_path.read_text())
    assert fm["recording_id"] == succeeded.short_id
    assert body.strip() == "[00:00] SPEAKER_00: Synthetic successful turn."
    clusters = read_sidecar(succeeded_path).clusters
    assert len(clusters) == 1
    assert clusters[0].speaker == "SPEAKER_00"
    assert clusters[0].centroid == pytest.approx([0.3, 0.9])
    assert clusters[0].speech_s == 2.35


def test_single_transcription_failure_leaves_no_success_artifacts(
    discovery_roots, monkeypatch, capsys
):
    import scribe

    import diane.transcribe
    from diane.db import canonical_transcript_path
    from diane.voiceprints import sidecar_path

    roots = discovery_roots
    monkeypatch.setattr(diane.transcribe, "RECORDINGS_DIR", roots.recordings)

    def reject(_audio, **kwargs):
        raise scribe.ScribeError("controlled diarization failure")

    monkeypatch.setattr(scribe, "run_nemotron_diarization", reject)
    monkeypatch.setattr(scribe, "run_transcription", lambda _audio: [])

    with pytest.raises(scribe.ScribeError, match="controlled diarization failure"):
        app(["transcribe", roots.pending.short_id], result_action="return_value")

    output = capsys.readouterr()
    assert "Wrote" not in output.out + output.err
    target = canonical_transcript_path(roots.pending, roots.transcripts)
    assert not target.exists()
    assert not sidecar_path(target).exists()


@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize("with_sidecar", [False, True])
def test_manual_blank_speaker_is_revisited_through_cli(
    discovery_roots, monkeypatch, capsys, bulk, with_sidecar,
):
    import diane.speakers
    from diane.db import canonical_transcript_path
    from diane.frontmatter import frontmatter_from_recording, parse_frontmatter
    from diane.voiceprints import Cluster, load_identity, new_identity, read_sidecar, sidecar_path

    roots = discovery_roots
    path = canonical_transcript_path(roots.pending, roots.transcripts)
    body = "[00:00] SPEAKER_00: Opening.\n[00:01] SPEAKER_01: Response.\n"
    if with_sidecar:
        new_identity(path, frontmatter_from_recording(roots.pending), body, [
            Cluster("SPEAKER_00", [1.0, 0.0], 1.0),
            Cluster("SPEAKER_01", [0.0, 1.0], 1.0),
        ]).save()
    else:
        path.write_text(body)
    command = ["speakers", "all" if bulk else roots.pending.short_id]
    before = path.read_bytes()
    sidecar_before = sidecar_path(path).read_bytes() if with_sidecar else None
    monkeypatch.setattr(diane.speakers, "prompt_with_snippets", lambda *args, **kwargs: "")
    app(command, result_action="return_value")
    assert path.read_bytes() == before
    assert len(load_identity(path, roots.pending).manual_targets) == 2
    if with_sidecar:
        assert sidecar_path(path).read_bytes() == sidecar_before
    else:
        assert not sidecar_path(path).exists()
    blank_output = capsys.readouterr()
    assert "Wrote" not in blank_output.out + blank_output.err
    first_answers = iter(["Synthetic Alpha", ""])
    monkeypatch.setattr(
        diane.speakers, "prompt_with_snippets", lambda *args, **kwargs: next(first_answers),
    )
    app(command, result_action="return_value")
    first = load_identity(path, roots.pending)
    assert [target.label for target in first.manual_targets] == ["SPEAKER_01"]
    assert first.frontmatter["speakers"] == ["Synthetic Alpha", "SPEAKER_01"]
    if with_sidecar:
        assert [(c.name, c.source, c.score) for c in read_sidecar(path).clusters] == [
            ("Synthetic Alpha", "manual", None), (None, None, None),
        ]
    else:
        assert not sidecar_path(path).exists()
    capsys.readouterr()
    prompted = []

    def finish(label, *args, **kwargs):
        prompted.append(label)
        return "Synthetic Beta"

    monkeypatch.setattr(diane.speakers, "prompt_with_snippets", finish)
    app(command, result_action="return_value")
    assert prompted == ["SPEAKER_01"]
    final = load_identity(path, roots.pending)
    assert not final.manual_targets
    assert final.frontmatter["speakers"] == ["Synthetic Alpha", "Synthetic Beta"]
    assert parse_frontmatter(path.read_text())[1].strip() == (
        "[00:00] Synthetic Alpha: Opening.\n[00:01] Synthetic Beta: Response."
    )
    if with_sidecar:
        assert all(c.source == "manual" and c.score is None for c in read_sidecar(path).clusters)
    else:
        assert not sidecar_path(path).exists()


def test_shared_audit_rejections_do_not_write_and_confirmation_remains_available(
    discovery_roots, monkeypatch, capsys,
):
    import diane.speakers
    from diane.db import canonical_transcript_path
    from diane.frontmatter import frontmatter_from_recording
    from diane.voiceprints import Cluster, load_identity, new_identity, read_sidecar, sidecar_path

    roots = discovery_roots
    seed_path = roots.transcripts / "seed.txt"
    seed = new_identity(seed_path, {}, "[00:00] SPEAKER_00: Seed.\n", [
        Cluster("SPEAKER_00", [1.0, 0.0], 1.0),
    ])
    assert seed.apply_manual({"SPEAKER_00": "Synthetic Shared"}) == []
    seed.save()
    path = canonical_transcript_path(roots.pending, roots.transcripts)
    document = new_identity(
        path, frontmatter_from_recording(roots.pending),
        "[00:00] SPEAKER_00: First.\n[00:01] SPEAKER_01: Second.\n",
        [Cluster("SPEAKER_00", [1.0, 0.0], 1.0), Cluster("SPEAKER_01", [1.0, 0.0], 1.0)],
    )
    document.save()
    before = (path.read_bytes(), sidecar_path(path).read_bytes())
    monkeypatch.setattr(
        diane.speakers, "prompt_with_snippets", lambda *args, **kwargs: "Synthetic Correction",
    )
    command = ["speakers", "audit", roots.pending.short_id]
    for _ in range(2):
        app(command, result_action="return_value")
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "ambiguous" in output.lower()
        assert "combine turns" in output
        assert "Saved" not in output
        assert before == (path.read_bytes(), sidecar_path(path).read_bytes())
        assert len(load_identity(path, roots.pending).audit_targets) == 2
    monkeypatch.setattr(diane.speakers, "prompt_with_snippets", lambda *args, **kwargs: "")
    app(command, result_action="return_value")
    assert not load_identity(path, roots.pending).audit_targets
    assert [(c.source, c.score) for c in read_sidecar(path).clusters] == [
        ("confirmed", 1.0), ("confirmed", 1.0),
    ]


def test_identity_partial_sidecar_write_never_reports_cli_success(
    discovery_roots, monkeypatch, capsys,
):
    import diane.speakers
    from diane.db import canonical_transcript_path
    from diane.frontmatter import frontmatter_from_recording, parse_frontmatter
    from diane.voiceprints import (
        Cluster, PartialIdentityWriteError, new_identity, sidecar_path,
    )

    roots = discovery_roots
    path = canonical_transcript_path(roots.pending, roots.transcripts)
    new_identity(
        path, frontmatter_from_recording(roots.pending),
        "[00:00] SPEAKER_00: Opening.\n", [Cluster("SPEAKER_00", [1.0, 0.0], 1.0)],
    ).save()

    def answer_during_storage_failure(*args, **kwargs):
        sidecar_path(path).unlink()
        sidecar_path(path).mkdir()
        return "Synthetic Named"

    monkeypatch.setattr(diane.speakers, "prompt_with_snippets", answer_during_storage_failure)
    with pytest.raises(PartialIdentityWriteError):
        app(["speakers", roots.pending.short_id], result_action="return_value")
    captured = capsys.readouterr()
    assert "Wrote" not in captured.out + captured.err
    assert "Saved" not in captured.out + captured.err
    fm, body = parse_frontmatter(path.read_text())
    assert fm["speakers"] == ["Synthetic Named"]
    assert "Synthetic Named: Opening." in body
    assert sidecar_path(path).is_dir()
