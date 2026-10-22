"""Tests for voiceprint sidecars, enrollment, and matching."""

import pytest

from diane.speakers import extract_speakers, replace_speakers, speaker_labels, unnamed_speakers
from diane.voiceprints import (
    Cluster,
    Sidecar,
    build_voiceprints,
    cosine,
    match,
    read_sidecar,
    sidecar_path,
    write_sidecar,
)


def _transcript(tmp_path, name):
    path = tmp_path / f"{name}.txt"
    path.write_text("---\n---\n")
    return path


def test_sidecar_round_trip(tmp_path):
    transcript = _transcript(tmp_path, "20260101T100000-aaaa")
    sidecar = Sidecar(clusters=[Cluster("SPEAKER_00", [1.0, 0.0], 12.5, "Practice A", "manual", None)])
    write_sidecar(transcript, sidecar)
    assert sidecar_path(transcript).name == "20260101T100000-aaaa.voice.json"
    assert read_sidecar(transcript) == sidecar


def test_missing_sidecar_is_none(tmp_path):
    assert read_sidecar(_transcript(tmp_path, "x")) is None


def test_voiceprints_use_only_manual_and_confirmed(tmp_path):
    a = _transcript(tmp_path, "a")
    write_sidecar(a, Sidecar(clusters=[
        Cluster("SPEAKER_00", [1.0, 0.0], 10, "Practice A", "manual"),
        Cluster("SPEAKER_01", [0.0, 1.0], 10, "Practice B", "auto", 0.9),
    ]))
    b = _transcript(tmp_path, "b")
    write_sidecar(b, Sidecar(clusters=[Cluster("SPEAKER_00", [0.0, 2.0], 10, "Practice C", "confirmed")]))
    voiceprints = build_voiceprints(tmp_path)
    assert set(voiceprints) == {"Practice A", "Practice C"}
    assert voiceprints["Practice C"] == [0.0, 1.0]


def test_voiceprints_can_exclude_a_transcript(tmp_path):
    a = _transcript(tmp_path, "a")
    write_sidecar(a, Sidecar(clusters=[Cluster("SPEAKER_00", [1.0, 0.0], 10, "Practice A", "manual")]))
    assert build_voiceprints(tmp_path, exclude=a) == {}


def test_voiceprint_is_mean_of_normalized_centroids(tmp_path):
    a = _transcript(tmp_path, "a")
    write_sidecar(a, Sidecar(clusters=[
        Cluster("SPEAKER_00", [10.0, 0.0], 10, "Practice A", "manual"),
        Cluster("SPEAKER_01", [0.0, 1.0], 10, "Practice A", "manual"),
    ]))
    vp = build_voiceprints(tmp_path)["Practice A"]
    assert abs(vp[0] - vp[1]) < 1e-9


def test_match_applies_threshold_and_allows_split_person():
    voiceprints = {"Practice A": [1.0, 0.0], "Practice B": [0.0, 1.0]}
    clusters = [
        Cluster("SPEAKER_00", [1.0, 0.05], 30),
        Cluster("SPEAKER_01", [0.98, 0.1], 5),
        Cluster("SPEAKER_02", [1.0, 1.0], 20),
    ]
    result = match(clusters, voiceprints, threshold=0.86)
    assert result["SPEAKER_00"][0] == "Practice A"
    assert result["SPEAKER_01"][0] == "Practice A"
    assert "SPEAKER_02" not in result  # cosine 0.707 to both: stays unnamed


def test_match_with_no_voiceprints_names_nothing():
    assert match([Cluster("SPEAKER_00", [1.0, 0.0], 30)], {}) == {}


def test_cosine():
    assert cosine([1.0, 0.0], [2.0, 0.0]) == 1.0
    assert abs(cosine([1.0, 0.0], [0.0, 3.0])) < 1e-12


def test_speaker_names_with_spaces_are_parsed():
    body = "[00:01] Practice A: Hello there.\n\n[00:05] Practice B: Hi: how are you?\n"
    assert extract_speakers(body) == ["Practice A", "Practice B"]


def test_unnamed_speakers_ignores_names():
    body = "[00:01] Practice A: Hi.\n\n[00:03] SPEAKER_01: Hello.\n\n[00:05] SPEAKER_02: Hey.\n"
    assert unnamed_speakers(body) == ["SPEAKER_01", "SPEAKER_02"]


def test_speaker_labels_preserve_order_and_repeated_occurrences():
    body = (
        "[00:01] Practice A: First: a colon in speech.\n\n"
        "[01:02:03] UNKNOWN: Unattributed.\n\n"
        "[2026-04-02T16:59:20] Practice A: Again.\n"
    )
    assert speaker_labels(body) == ["Practice A", "UNKNOWN", "Practice A"]
    assert extract_speakers(body) == ["Practice A", "UNKNOWN"]


def test_speaker_replacements_are_simultaneous_and_literal():
    body = (
        "[00:01] A: B: is text, not another label.\n\n"
        "[01:02:03] B: Original B.\n\n"
        "[2026-04-02T16:59:20] A: Original A again.\n"
    )
    assert replace_speakers(body, {"A": "B", "B": "C"}) == (
        "[00:01] B: B: is text, not another label.\n\n"
        "[01:02:03] C: Original B.\n\n"
        "[2026-04-02T16:59:20] B: Original A again.\n"
    )


def test_voiceprints_preserve_orphan_sidecar_enrollment(tmp_path):
    transcript = _transcript(tmp_path, "removed")
    write_sidecar(transcript, Sidecar(clusters=[
        Cluster("SPEAKER_00", [1.0, 0.0], 0.01, "Practice A", "manual"),
    ]))
    transcript.unlink()
    assert build_voiceprints(tmp_path) == {"Practice A": [1.0, 0.0]}


def test_voiceprint_centroids_are_equal_weight_not_speech_weighted(tmp_path):
    transcript = _transcript(tmp_path, "unequal-duration")
    write_sidecar(transcript, Sidecar(clusters=[
        Cluster("SPEAKER_00", [100.0, 0.0], 1000, "Practice A", "manual"),
        Cluster("SPEAKER_01", [0.0, 1.0], 0.01, "Practice A", "confirmed"),
        Cluster("SPEAKER_02", [1.0, 0.0], 1000, None, "manual"),
        Cluster("SPEAKER_03", [1.0, 0.0], 1000, "", "confirmed"),
        Cluster("SPEAKER_04", [1.0, 0.0], 1000, "OTHER", None),
    ]))
    assert build_voiceprints(tmp_path) == {"Practice A": pytest.approx([2 ** -0.5, 2 ** -0.5])}


def test_match_includes_exact_threshold_and_rounds_score():
    clusters = [Cluster("SPEAKER_00", [0.86, (1 - 0.86 ** 2) ** 0.5], 0.01)]
    assert match(clusters, {"Practice A": [1.0, 0.0]}) == {"SPEAKER_00": ("Practice A", 0.86)}
    assert match(clusters, {"Practice A": [1.0, 0.0]}, threshold=0.8601) == {}
    assert match(
        [Cluster("SPEAKER_01", [1.0, 0.05], 0.01)], {"Practice A": [1.0, 0.0]},
    ) == {"SPEAKER_01": ("Practice A", 0.9988)}


@pytest.mark.parametrize("source", ["manual", "confirmed"])
def test_legacy_anonymous_placeholder_name_never_enrolls(tmp_path, source):
    transcript = _transcript(tmp_path, "legacy")
    write_sidecar(transcript, Sidecar(clusters=[
        Cluster("SPEAKER_00", [1.0, 0.0], 10, "SPEAKER_00", source),
        Cluster("SPEAKER_01", [1.0, 0.0], 10, "SPEAKER_99", source),
        Cluster("SPEAKER_02", [0.0, 1.0], 0.01, "Named Person", source),
    ]))
    assert build_voiceprints(tmp_path) == {"Named Person": [0.0, 1.0]}
