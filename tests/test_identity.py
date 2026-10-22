"""Identity transitions through the public lifecycle and real transcript/sidecar files."""

import json
from dataclasses import FrozenInstanceError, asdict
from datetime import datetime, timedelta, timezone

import pytest

from diane.db import Recording
from diane.frontmatter import build_output, parse_frontmatter
from diane.voiceprints import (
    Cluster,
    PartialIdentityWriteError,
    Sidecar,
    build_voiceprints,
    load_identity,
    new_identity,
    read_sidecar,
    sidecar_path,
    write_sidecar,
)


METADATA = {
    "recording_id": "01234567",
    "recorded": "2026-04-05T14:00:00+00:00",
    "duration": 123.5,
    "title": "Private planning",
    "custom": {"keep": ["unaltered", 7]},
}
RAW_BODY = "[00:01] SPEAKER_00: First: keep punctuation.\n\n[00:05] SPEAKER_01: Second.\n"


def _enrolled(tmp_path):
    directory = tmp_path / "enrolled"
    directory.mkdir(exist_ok=True)
    doc = new_identity(
        directory / "seed.txt", {}, RAW_BODY,
        [Cluster("SPEAKER_00", [1.0, 0.0], 10), Cluster("SPEAKER_01", [0.0, 1.0], 2)],
    )
    assert doc.apply_manual({"SPEAKER_00": "A", "SPEAKER_01": "B"}) == []
    assert doc.save() is True
    return directory


def _automatic(tmp_path, *, body=RAW_BODY, clusters=None):
    if clusters is None:
        clusters = [Cluster("SPEAKER_00", [1.0, 0.05], 12.5), Cluster("SPEAKER_01", [0.0, 1.0], 3)]
    doc = new_identity(
        tmp_path / "target.txt", dict(METADATA), body, clusters,
        transcripts_dir=_enrolled(tmp_path),
    )
    assert doc.save() is True
    return load_identity(doc.path)


def _snapshot(path):
    paths = [path, sidecar_path(path)]
    return {item: (item.read_bytes(), item.stat().st_mtime_ns) for item in paths if item.exists()}


def _persisted(doc, expected_body, expected_speakers):
    fm, body = parse_frontmatter(doc.path.read_text())
    assert body == expected_body
    assert fm["speakers"] == expected_speakers
    for key, value in METADATA.items():
        assert fm[key] == value
    assert read_sidecar(doc.path) == doc.sidecar


def _remove_provenance(path):
    storage = sidecar_path(path)
    data = json.loads(storage.read_text())
    data.pop("provenance", None)
    storage.write_text(json.dumps(data))


def _assert_rejected(rejections, keys):
    assert {item.key for item in rejections} == set(keys)
    assert all(item.label and item.reason for item in rejections)


def test_new_identity_stages_auto_names_without_enrollment_or_early_writes(tmp_path):
    enrollment = _enrolled(tmp_path)
    path = tmp_path / "target.txt"
    doc = new_identity(
        path, dict(METADATA), RAW_BODY,
        [Cluster("SPEAKER_00", [1.0, 0.05], 12.5), Cluster("SPEAKER_01", [1.0, 1.0], 3)],
        transcripts_dir=enrollment,
    )
    assert doc.path == path
    assert doc.matches == {"SPEAKER_00": ("A", 0.9988)}
    assert not path.exists()
    assert not sidecar_path(path).exists()
    assert [(target.key, target.label) for target in doc.manual_targets] == [("SPEAKER_01", "SPEAKER_01")]
    assert [(target.key, target.label, target.score, target.speech_s, target.shared) for target in doc.audit_targets] == [
        ("SPEAKER_00", "A", 0.9988, 12.5, False),
    ]
    assert doc.save() is True
    _persisted(doc, RAW_BODY.replace("SPEAKER_00", "A"), ["A", "SPEAKER_01"])
    assert build_voiceprints(tmp_path) == {}
    assert [(cluster.name, cluster.source, cluster.score) for cluster in read_sidecar(path).clusters] == [
        ("A", "auto", 0.9988), (None, None, None),
    ]


def test_initial_matching_excludes_target_transcript_and_uses_parent_by_default(tmp_path):
    path = tmp_path / "target.txt"
    prior = new_identity(path, {}, "[00:01] SPEAKER_00: Prior.\n", [Cluster("SPEAKER_00", [1.0, 0.0], 10)])
    prior.apply_manual({"SPEAKER_00": "A"})
    prior.save()
    replacement = new_identity(path, {}, "[00:01] SPEAKER_00: New.\n", [Cluster("SPEAKER_00", [1.0, 0.0], 10)])
    assert replacement.matches == {}
    assert [target.label for target in replacement.manual_targets] == ["SPEAKER_00"]
    other = new_identity(tmp_path / "other.txt", {}, "[00:01] SPEAKER_00: Other.\n", [Cluster("SPEAKER_00", [1.0, 0.0], 10)])
    assert other.matches == {"SPEAKER_00": ("A", 1.0)}


@pytest.mark.parametrize("answer", ["", "A"])
def test_audit_confirmation_retains_score_and_enrolls_only_explicit_target(tmp_path, answer):
    doc = _automatic(tmp_path)
    original_body = doc.body
    assert doc.apply_audit({"SPEAKER_00": answer}) == []
    assert doc.save() is True
    _persisted(doc, original_body, ["A", "B"])
    clusters = read_sidecar(doc.path).clusters
    assert (clusters[0].name, clusters[0].source, clusters[0].score) == ("A", "confirmed", 0.9988)
    assert (clusters[1].name, clusters[1].source, clusters[1].score) == ("B", "auto", 1.0)
    assert set(build_voiceprints(tmp_path)) == {"A"}
    repeated = load_identity(doc.path)
    assert [target.key for target in repeated.audit_targets] == ["SPEAKER_01"]
    before = _snapshot(doc.path)
    assert repeated.apply_audit({}) == []
    assert repeated.save() is False
    assert _snapshot(doc.path) == before


def test_unique_audit_correction_updates_body_frontmatter_and_enrollment(tmp_path):
    doc = _automatic(tmp_path)
    assert doc.apply_audit({"SPEAKER_00": "Correct Person"}) == []
    assert doc.save() is True
    _persisted(doc, RAW_BODY.replace("SPEAKER_00", "Correct Person").replace("SPEAKER_01", "B"), ["Correct Person", "B"])
    corrected, untouched = read_sidecar(doc.path).clusters
    assert (corrected.name, corrected.source, corrected.score) == ("Correct Person", "manual", None)
    assert (untouched.name, untouched.source, untouched.score) == ("B", "auto", 1.0)
    assert set(build_voiceprints(tmp_path)) == {"Correct Person"}
    assert [target.key for target in load_identity(doc.path).audit_targets] == ["SPEAKER_01"]


def test_blank_manual_answer_preserves_cluster_then_revisit_names_remaining_body_label(tmp_path):
    doc = new_identity(tmp_path / "target.txt", dict(METADATA), RAW_BODY, [
        Cluster("SPEAKER_00", [1.0, 0.0], 12.5), Cluster("SPEAKER_01", [0.0, 1.0], 3),
    ])
    doc.apply_manual({"SPEAKER_00": "A", "SPEAKER_01": ""})
    assert doc.save() is True
    _persisted(doc, RAW_BODY.replace("SPEAKER_00", "A"), ["A", "SPEAKER_01"])
    skipped = asdict(read_sidecar(doc.path).clusters[1])
    assert (skipped["name"], skipped["source"], skipped["score"]) == (None, None, None)
    revisit = load_identity(doc.path)
    assert [(target.key, target.label) for target in revisit.manual_targets] == [("SPEAKER_01", "SPEAKER_01")]
    before = _snapshot(doc.path)
    assert revisit.apply_manual({"SPEAKER_01": ""}) == []
    assert revisit.save() is False
    assert _snapshot(doc.path) == before
    assert asdict(read_sidecar(doc.path).clusters[1]) == skipped
    assert revisit.apply_manual({"SPEAKER_01": "B"}) == []
    assert revisit.save() is True
    _persisted(revisit, RAW_BODY.replace("SPEAKER_00", "A").replace("SPEAKER_01", "B"), ["A", "B"])
    assert load_identity(doc.path).manual_targets == []
    assert set(build_voiceprints(tmp_path)) == {"A", "B"}


def test_existing_all_blank_manual_pass_does_not_normalize_frontmatter(tmp_path):
    path = tmp_path / "legacy.txt"
    original = "---\ntitle: keep exact YAML\nspeakers: [ALREADY]\n---\n" + RAW_BODY
    path.write_text(original)
    doc = load_identity(path)
    before = _snapshot(path)
    assert [target.key for target in doc.manual_targets] == ["SPEAKER_00", "SPEAKER_01"]
    assert doc.apply_manual({"SPEAKER_00": "", "SPEAKER_01": ""}) == []
    assert doc.save() is False
    assert _snapshot(path) == before
    assert path.read_text() == original


def test_missing_sidecar_manual_naming_is_transcript_only_without_invented_embeddings(tmp_path):
    path = tmp_path / "legacy.txt"
    path.write_text(build_output(dict(METADATA), RAW_BODY))
    doc = load_identity(path)
    assert doc.sidecar is None
    assert doc.audit_targets == []
    assert [(target.key, target.score, target.speech_s) for target in doc.manual_targets] == [
        ("SPEAKER_00", None, 0.0), ("SPEAKER_01", None, 0.0),
    ]
    assert doc.apply_manual({"SPEAKER_00": "A"}) == []
    assert doc.save() is True
    fm, body = parse_frontmatter(path.read_text())
    assert fm == {**METADATA, "speakers": ["A", "SPEAKER_01"]}
    assert body == RAW_BODY.replace("SPEAKER_00", "A")
    assert not sidecar_path(path).exists()
    assert [target.key for target in load_identity(path).manual_targets] == ["SPEAKER_01"]


def test_legacy_recording_metadata_is_staged_without_preflight_write(tmp_path):
    path = tmp_path / "legacy.txt"
    path.write_text(RAW_BODY)
    recording = Recording("0123456789", datetime(2026, 4, 5, 14, tzinfo=timezone.utc), timedelta(seconds=123.5), "Planning", "/unused/private.m4a")
    before = _snapshot(path)
    doc = load_identity(path, recording=recording)
    assert _snapshot(path) == before
    assert doc.apply_manual({"SPEAKER_00": ""}) == []
    assert doc.save() is False
    assert _snapshot(path) == before
    assert doc.apply_manual({"SPEAKER_00": "A"}) == []
    assert doc.save() is True
    fm, body = parse_frontmatter(path.read_text())
    assert fm["recording_id"] == "01234567"
    assert fm["title"] == "Planning"
    assert fm["duration"] == 123.5
    assert datetime.fromisoformat(fm["recorded"]) == recording.date
    assert body == RAW_BODY.replace("SPEAKER_00", "A")
    assert not sidecar_path(path).exists()


@pytest.mark.parametrize("storage", ["malformed-json", "directory"])
def test_invalid_sidecar_is_explicit_error_not_missing(tmp_path, storage):
    path = tmp_path / "legacy.txt"
    path.write_text(RAW_BODY)
    if storage == "malformed-json":
        sidecar_path(path).write_text("{not json")
        error = json.JSONDecodeError
    else:
        sidecar_path(path).mkdir()
        error = OSError
    original = path.read_bytes()
    with pytest.raises(error):
        load_identity(path)
    assert path.read_bytes() == original


def test_targets_and_rejections_are_immutable_original_keys(tmp_path):
    doc = _automatic(tmp_path)
    target = doc.audit_targets[0]
    with pytest.raises(FrozenInstanceError):
        target.key = "A"
    _remove_provenance(doc.path)
    legacy = load_identity(doc.path)
    rejection = legacy.apply_audit({"SPEAKER_00": "C"})[0]
    assert (rejection.key, rejection.label) == ("SPEAKER_00", "A")
    with pytest.raises(FrozenInstanceError):
        rejection.reason = "different"


def test_shared_auto_name_rejects_both_corrections_and_successive_attempts_without_writes(tmp_path):
    doc = _automatic(tmp_path, clusters=[Cluster("SPEAKER_00", [1.0, 0.0], 10), Cluster("SPEAKER_01", [1.0, 0.0], 2)])
    assert [(target.key, target.label, target.shared) for target in doc.audit_targets] == [
        ("SPEAKER_00", "A", True), ("SPEAKER_01", "A", True),
    ]
    before = _snapshot(doc.path)
    clusters_before = [asdict(cluster) for cluster in doc.sidecar.clusters]
    _assert_rejected(doc.apply_audit({"SPEAKER_00": "C", "SPEAKER_01": "D"}), ["SPEAKER_00", "SPEAKER_01"])
    assert [asdict(cluster) for cluster in doc.sidecar.clusters] == clusters_before
    assert doc.save() is False
    assert _snapshot(doc.path) == before
    _assert_rejected(doc.apply_audit({"SPEAKER_01": "D"}), ["SPEAKER_01"])
    assert doc.save() is False
    assert _snapshot(doc.path) == before
    assert {target.key for target in load_identity(doc.path).audit_targets} == {"SPEAKER_00", "SPEAKER_01"}


def test_shared_auto_name_can_be_confirmed_without_relabeling(tmp_path):
    doc = _automatic(tmp_path, clusters=[Cluster("SPEAKER_00", [1.0, 0.0], 10), Cluster("SPEAKER_01", [1.0, 0.0], 2)])
    body = doc.body
    assert doc.apply_audit({"SPEAKER_00": "", "SPEAKER_01": "A"}) == []
    assert doc.save() is True
    _persisted(doc, body, ["A"])
    assert [(cluster.name, cluster.source, cluster.score) for cluster in doc.sidecar.clusters] == [
        ("A", "confirmed", 1.0), ("A", "confirmed", 1.0),
    ]
    assert load_identity(doc.path).audit_targets == []
    assert set(build_voiceprints(tmp_path)) == {"A"}


def test_rejected_shared_group_stays_unchanged_when_other_decisions_are_saved(tmp_path):
    body = RAW_BODY + "\n[00:09] SPEAKER_02: Independent.\n"
    doc = _automatic(tmp_path, body=body, clusters=[
        Cluster("SPEAKER_00", [1.0, 0.0], 10), Cluster("SPEAKER_01", [1.0, 0.0], 2),
        Cluster("SPEAKER_02", [0.0, 1.0], 3),
    ])
    original_shared = [asdict(cluster) for cluster in doc.sidecar.clusters[:2]]
    _assert_rejected(doc.apply_audit({"SPEAKER_00": "C", "SPEAKER_01": "D", "SPEAKER_02": "E"}), ["SPEAKER_00", "SPEAKER_01"])
    assert doc.save() is True
    _persisted(doc, body.replace("SPEAKER_00", "A").replace("SPEAKER_01", "A").replace("SPEAKER_02", "E"), ["A", "E"])
    assert [asdict(cluster) for cluster in read_sidecar(doc.path).clusters[:2]] == original_shared
    assert {target.key for target in load_identity(doc.path).audit_targets} == {"SPEAKER_00", "SPEAKER_01"}
    assert set(build_voiceprints(tmp_path)) == {"E"}


def test_unique_replacements_are_simultaneous_not_cascading(tmp_path):
    doc = _automatic(tmp_path)
    assert doc.apply_audit({"SPEAKER_00": "B", "SPEAKER_01": "C"}) == []
    assert doc.save() is True
    _persisted(doc, RAW_BODY.replace("SPEAKER_00", "B").replace("SPEAKER_01", "C"), ["B", "C"])
    assert [(cluster.name, cluster.source, cluster.score) for cluster in doc.sidecar.clusters] == [
        ("B", "manual", None), ("C", "manual", None),
    ]


@pytest.mark.parametrize("reload", [False, True])
def test_created_shared_name_makes_later_independent_correction_ambiguous(tmp_path, reload):
    doc = _automatic(tmp_path)
    assert doc.apply_audit({"SPEAKER_00": "B"}) == []
    assert doc.save() is True
    if reload:
        doc = load_identity(doc.path)
    before = _snapshot(doc.path)
    assert [(target.key, target.shared) for target in doc.audit_targets] == [("SPEAKER_01", True)]
    _assert_rejected(doc.apply_audit({"SPEAKER_01": "C"}), ["SPEAKER_01"])
    assert doc.save() is False
    assert _snapshot(doc.path) == before
    assert [(cluster.name, cluster.source) for cluster in doc.sidecar.clusters] == [("B", "manual"), ("B", "auto")]


@pytest.mark.parametrize("remove_body_only_mapping", [False, True])
def test_body_only_label_collision_is_shared_even_without_embedding(tmp_path, remove_body_only_mapping):
    body = "[00:01] SPEAKER_00: Embedded.\n\n[00:05] A: No embedding.\n\n[00:09] UNKNOWN: Unknown attribution.\n"
    doc = _automatic(tmp_path, body=body, clusters=[Cluster("SPEAKER_00", [1.0, 0.0], 10)])
    assert len(doc.sidecar.clusters) == 1
    data = json.loads(sidecar_path(doc.path).read_text())
    assert data["provenance"]["labels"] == {"SPEAKER_00": "A", "A": "A", "UNKNOWN": "UNKNOWN"}
    if remove_body_only_mapping:
        del data["provenance"]["labels"]["A"]
        sidecar_path(doc.path).write_text(json.dumps(data))
    doc = load_identity(doc.path)
    before = _snapshot(doc.path)
    _assert_rejected(doc.apply_audit({"SPEAKER_00": "C"}), ["SPEAKER_00"])
    assert doc.save() is False
    assert _snapshot(doc.path) == before
    assert len(read_sidecar(doc.path).clusters) == 1
    assert doc.apply_audit({"SPEAKER_00": ""}) == []
    assert doc.save() is True
    _persisted(doc, body.replace("SPEAKER_00", "A"), ["A", "UNKNOWN"])
    assert len(read_sidecar(doc.path).clusters) == 1


@pytest.mark.parametrize("damage", ["absent", "partial", "inconsistent", "version", "fingerprint", "stale-body"])
def test_unknown_provenance_rejects_even_single_cluster_correction_but_allows_confirmation(tmp_path, damage):
    body = "[00:01] SPEAKER_00: Named.\n\n[00:05] UNKNOWN: No embedding.\n"
    doc = _automatic(tmp_path, body=body, clusters=[Cluster("SPEAKER_00", [1.0, 0.0], 10)])
    data = json.loads(sidecar_path(doc.path).read_text())
    if damage == "absent":
        del data["provenance"]
    elif damage == "partial":
        del data["provenance"]["labels"]["UNKNOWN"]
    elif damage == "inconsistent":
        data["provenance"]["labels"]["SPEAKER_00"] = "B"
    elif damage == "version":
        data["provenance"]["version"] = 999
    elif damage == "fingerprint":
        data["provenance"]["fingerprint"] = "not-valid"
    else:
        fm, _ = parse_frontmatter(doc.path.read_text())
        doc.path.write_text(build_output(fm, "[00:01] UNKNOWN: No embedding.\n\n[00:05] A: Named.\n"))
    sidecar_path(doc.path).write_text(json.dumps(data))
    doc = load_identity(doc.path)
    before = _snapshot(doc.path)
    _assert_rejected(doc.apply_audit({"SPEAKER_00": "C"}), ["SPEAKER_00"])
    assert doc.save() is False
    assert _snapshot(doc.path) == before
    assert (doc.sidecar.clusters[0].name, doc.sidecar.clusters[0].source, doc.sidecar.clusters[0].score) == ("A", "auto", 1.0)
    assert doc.apply_audit({"SPEAKER_00": "A"}) == []
    assert doc.save() is True
    reloaded = load_identity(doc.path)
    assert reloaded.audit_targets == []
    assert (reloaded.sidecar.clusters[0].name, reloaded.sidecar.clusters[0].source, reloaded.sidecar.clusters[0].score) == ("A", "confirmed", 1.0)


@pytest.mark.parametrize("source", ["auto", "manual"])
def test_hand_edited_raw_turn_cannot_rename_existing_named_cluster(tmp_path, source):
    raw_body = "[00:01] SPEAKER_00: First turn.\n\n[00:05] SPEAKER_00: Second turn.\n"
    clusters = [Cluster("SPEAKER_00", [1.0, 0.05], 12.5)]
    if source == "auto":
        doc = _automatic(tmp_path, body=raw_body, clusters=clusters)
        score = 0.9988
    else:
        doc = new_identity(tmp_path / "target.txt", dict(METADATA), raw_body, clusters)
        assert doc.apply_manual({"SPEAKER_00": "A"}) == []
        assert doc.save() is True
        score = None
    fm, body = parse_frontmatter(doc.path.read_text())
    assert body == raw_body.replace("SPEAKER_00", "A")
    edited_body = body.replace("] A:", "] SPEAKER_00:", 1)
    doc.path.write_text(build_output(fm, edited_body))
    doc = load_identity(doc.path)
    assert [(target.key, target.label) for target in doc.manual_targets] == [
        ("SPEAKER_00", "SPEAKER_00"),
    ]
    before = _snapshot(doc.path)
    original_sidecar = asdict(doc.sidecar)
    original_audit_targets = doc.audit_targets
    rejections = doc.apply_manual({"SPEAKER_00": "C"})
    _assert_rejected(rejections, ["SPEAKER_00"])
    assert "ambiguous" in rejections[0].reason
    assert doc.body == edited_body
    assert doc.frontmatter == fm
    assert asdict(doc.sidecar) == original_sidecar
    assert (doc.sidecar.clusters[0].name, doc.sidecar.clusters[0].source, doc.sidecar.clusters[0].score) == ("A", source, score)
    assert doc.audit_targets == original_audit_targets
    assert doc.save() is False
    assert _snapshot(doc.path) == before
    reloaded = load_identity(doc.path)
    assert reloaded.body == edited_body
    assert asdict(reloaded.sidecar) == original_sidecar
    if source == "auto":
        assert [(target.key, target.label, target.score) for target in reloaded.audit_targets] == [
            ("SPEAKER_00", "A", score),
        ]
        assert reloaded.apply_audit({"SPEAKER_00": ""}) == []
        assert reloaded.save() is True
        confirmed = load_identity(doc.path)
        _persisted(confirmed, edited_body, ["SPEAKER_00", "A"])
        assert (confirmed.sidecar.clusters[0].name, confirmed.sidecar.clusters[0].source, confirmed.sidecar.clusters[0].score) == ("A", "confirmed", score)
        assert confirmed.sidecar.provenance == original_sidecar["provenance"]
        assert confirmed.audit_targets == []
    else:
        assert reloaded.audit_targets == []


def test_timestamp_and_text_only_edits_keep_provenance_valid(tmp_path):
    doc = _automatic(tmp_path)
    original_provenance = json.loads(sidecar_path(doc.path).read_text())["provenance"]
    fm, body = parse_frontmatter(doc.path.read_text())
    absolute = body.replace("[00:01]", "[2026-04-05T14:00:01]").replace("[00:05]", "[2026-04-05T14:00:05]").replace("Second.", "Corrected transcription text.")
    fm["timestamps"] = "absolute"
    doc.path.write_text(build_output(fm, absolute))
    doc = load_identity(doc.path)
    assert json.loads(sidecar_path(doc.path).read_text())["provenance"] == original_provenance
    assert doc.apply_audit({"SPEAKER_00": "C"}) == []
    assert doc.save() is True
    _persisted(doc, absolute.replace("] A:", "] C:"), ["C", "B"])
    assert parse_frontmatter(doc.path.read_text())[0]["timestamps"] == "absolute"


@pytest.mark.parametrize("with_sidecar", [False, True])
def test_legacy_raw_manual_label_can_change_without_fabricating_provenance(tmp_path, with_sidecar):
    path = tmp_path / "legacy.txt"
    path.write_text(build_output(dict(METADATA), RAW_BODY))
    if with_sidecar:
        write_sidecar(path, Sidecar(clusters=[Cluster("SPEAKER_00", [1.0, 0.0], 10)]))
    doc = load_identity(path)
    assert doc.apply_manual({"SPEAKER_00": "A"}) == []
    assert doc.save() is True
    assert load_identity(path).body == RAW_BODY.replace("SPEAKER_00", "A")
    if with_sidecar:
        data = json.loads(sidecar_path(path).read_text())
        assert data.get("provenance") is None
        assert len(data["clusters"]) == 1
        assert (data["clusters"][0]["name"], data["clusters"][0]["source"], data["clusters"][0]["score"]) == ("A", "manual", None)
    else:
        assert not sidecar_path(path).exists()
    # The remaining raw label has no matching cluster, even beside a named one.
    revisit = load_identity(path)
    assert [(target.key, target.label) for target in revisit.manual_targets] == [
        ("SPEAKER_01", "SPEAKER_01"),
    ]
    assert revisit.apply_manual({"SPEAKER_01": "B"}) == []
    assert revisit.save() is True
    assert load_identity(path).body == RAW_BODY.replace("SPEAKER_00", "A").replace("SPEAKER_01", "B")
    assert read_sidecar(path) == doc.sidecar


@pytest.mark.parametrize("with_sidecar", [False, True])
def test_legacy_raw_manual_merge_into_existing_display_is_rejected(tmp_path, with_sidecar):
    path = tmp_path / "legacy.txt"
    body = "[00:01] A: Existing name without provenance.\n\n[00:05] SPEAKER_01: Raw.\n"
    path.write_text(body)
    if with_sidecar:
        write_sidecar(path, Sidecar(clusters=[Cluster("SPEAKER_01", [0.0, 1.0], 10)]))
    before = _snapshot(path)
    doc = load_identity(path)
    _assert_rejected(doc.apply_manual({"SPEAKER_01": "A"}), ["SPEAKER_01"])
    assert doc.save() is False
    assert _snapshot(path) == before


def test_legacy_same_pass_untracked_manual_merge_rejects_both_answers(tmp_path):
    path = tmp_path / "legacy.txt"
    path.write_text(RAW_BODY)
    doc = load_identity(path)
    before = _snapshot(path)
    _assert_rejected(doc.apply_manual({"SPEAKER_00": "A", "SPEAKER_01": "A"}), ["SPEAKER_00", "SPEAKER_01"])
    assert doc.save() is False
    assert _snapshot(path) == before


def test_untracked_legacy_merge_rejection_does_not_block_independent_manual_answer(tmp_path):
    path = tmp_path / "legacy.txt"
    body = "[00:00] A: Existing.\n\n" + RAW_BODY
    path.write_text(build_output(dict(METADATA), body))
    doc = load_identity(path)
    _assert_rejected(doc.apply_manual({"SPEAKER_00": "A", "SPEAKER_01": "B"}), ["SPEAKER_00"])
    assert doc.save() is True
    fm, persisted = parse_frontmatter(path.read_text())
    assert persisted == body.replace("SPEAKER_01", "B")
    assert fm["speakers"] == ["A", "SPEAKER_00", "B"]
    assert not sidecar_path(path).exists()


def test_sidecar_metadata_and_cluster_audio_data_survive_identity_changes(tmp_path):
    doc = _automatic(tmp_path)
    data = json.loads(sidecar_path(doc.path).read_text())
    data.update(embedding_model="local-test-centroids", diarizer="original-diarizer", origin="imported")
    sidecar_path(doc.path).write_text(json.dumps(data))
    original_clusters = [(cluster.speaker, list(cluster.centroid), cluster.speech_s) for cluster in read_sidecar(doc.path).clusters]
    doc = load_identity(doc.path)
    assert doc.apply_audit({"SPEAKER_00": "C", "SPEAKER_01": ""}) == []
    assert doc.save() is True
    stored = read_sidecar(doc.path)
    assert (stored.embedding_model, stored.diarizer, stored.origin) == ("local-test-centroids", "original-diarizer", "imported")
    assert [(cluster.speaker, cluster.centroid, cluster.speech_s) for cluster in stored.clusters] == original_clusters


def test_sidecar_failure_reports_explicit_partial_write_after_transcript_success(tmp_path):
    doc = _automatic(tmp_path)
    original_body = doc.body
    assert doc.apply_audit({"SPEAKER_00": "C"}) == []
    storage = sidecar_path(doc.path)
    storage.unlink()
    storage.mkdir()
    with pytest.raises(PartialIdentityWriteError) as raised:
        doc.save()
    assert raised.value.transcript_path == doc.path
    assert raised.value.sidecar_path == storage
    assert isinstance(raised.value.__cause__, OSError)
    fm, persisted = parse_frontmatter(doc.path.read_text())
    assert persisted == original_body.replace("] A:", "] C:")
    assert fm["speakers"] == ["C", "B"]
    assert storage.is_dir()


def test_same_pass_confirmation_does_not_make_shared_sibling_correction_safe(tmp_path):
    doc = _automatic(tmp_path, clusters=[
        Cluster("SPEAKER_00", [1.0, 0.0], 10), Cluster("SPEAKER_01", [1.0, 0.0], 2),
    ])
    original_body = doc.body
    sibling_before = asdict(doc.sidecar.clusters[1])
    _assert_rejected(
        doc.apply_audit({"SPEAKER_00": "", "SPEAKER_01": "C"}),
        ["SPEAKER_01"],
    )
    assert doc.save() is True
    _persisted(doc, original_body, ["A"])
    assert doc.sidecar.clusters[0].source == "confirmed"
    assert asdict(read_sidecar(doc.path).clusters[1]) == sibling_before
    assert [target.key for target in load_identity(doc.path).audit_targets] == ["SPEAKER_01"]


def test_body_only_anonymous_manual_merge_remains_proven_shared_after_reload(tmp_path):
    doc = _automatic(tmp_path, clusters=[Cluster("SPEAKER_00", [1.0, 0.0], 10)])
    assert [(target.key, target.label) for target in doc.manual_targets] == [
        ("SPEAKER_01", "SPEAKER_01"),
    ]
    assert doc.apply_manual({"SPEAKER_01": "A"}) == []
    assert doc.save() is True
    doc = load_identity(doc.path)
    data = json.loads(sidecar_path(doc.path).read_text())
    assert data["provenance"]["labels"] == {"SPEAKER_00": "A", "SPEAKER_01": "A"}
    assert len(data["clusters"]) == 1
    assert [(target.key, target.shared) for target in doc.audit_targets] == [("SPEAKER_00", True)]
    before = _snapshot(doc.path)
    _assert_rejected(doc.apply_audit({"SPEAKER_00": "C"}), ["SPEAKER_00"])
    assert doc.save() is False
    assert _snapshot(doc.path) == before
    assert load_identity(doc.path).body == RAW_BODY.replace("SPEAKER_00", "A").replace("SPEAKER_01", "A")


def test_rejected_legacy_audit_does_not_write_missing_recording_frontmatter(tmp_path):
    path = tmp_path / "legacy.txt"
    path.write_text("[00:01] A: Legacy named turn.\n")
    write_sidecar(path, Sidecar(clusters=[
        Cluster("SPEAKER_00", [1.0, 0.0], 10, "A", "auto", 0.91),
    ]))
    recording = Recording(
        "0123456789", datetime(2026, 4, 5, 14, tzinfo=timezone.utc),
        timedelta(seconds=123.5), "Planning", "/unused/private.m4a",
    )
    before = _snapshot(path)
    doc = load_identity(path, recording=recording)
    _assert_rejected(doc.apply_audit({"SPEAKER_00": "C"}), ["SPEAKER_00"])
    assert doc.save() is False
    assert _snapshot(path) == before


def test_provenance_includes_actual_cluster_absent_from_body_without_new_body_turns(tmp_path):
    body = "[00:01] SPEAKER_00: Present.\n\n[00:05] UNKNOWN: Unattributed.\n"
    doc = new_identity(tmp_path / "target.txt", {}, body, [
        Cluster("SPEAKER_00", [1.0, 0.0], 10),
        Cluster("SPEAKER_99", [0.0, 1.0], 2),
    ])
    assert doc.save() is True
    data = json.loads(sidecar_path(doc.path).read_text())
    assert data["provenance"]["labels"] == {
        "SPEAKER_00": "SPEAKER_00", "SPEAKER_99": "SPEAKER_99", "UNKNOWN": "UNKNOWN",
    }
    assert load_identity(doc.path).body == body
    assert len(read_sidecar(doc.path).clusters) == 2


def test_blank_manual_preserves_existing_score_and_later_name_clears_it(tmp_path):
    path = tmp_path / "legacy.txt"
    body = "[00:01] SPEAKER_00: Raw label with prior metadata.\n"
    path.write_text(body)
    write_sidecar(path, Sidecar(clusters=[
        Cluster("SPEAKER_00", [1.0, 0.0], 0.01, score=0.72),
    ]))
    doc = load_identity(path)
    before = _snapshot(path)
    assert doc.apply_manual({"SPEAKER_00": ""}) == []
    assert doc.save() is False
    assert _snapshot(path) == before
    assert doc.sidecar.clusters[0].score == 0.72
    assert doc.apply_manual({"SPEAKER_00": "A"}) == []
    assert doc.save() is True
    cluster = read_sidecar(path).clusters[0]
    assert (cluster.name, cluster.source, cluster.score) == ("A", "manual", None)
    assert (cluster.speaker, cluster.centroid, cluster.speech_s) == ("SPEAKER_00", [1.0, 0.0], 0.01)


@pytest.mark.parametrize("workflow", ["manual", "audit"])
def test_five_word_name_roundtrips_through_public_lifecycle(tmp_path, workflow):
    if workflow == "audit":
        doc = _automatic(tmp_path)
        original_label = "A"
        other_label = "B"
        apply = doc.apply_audit
    else:
        doc = new_identity(tmp_path / "target.txt", dict(METADATA), RAW_BODY, [
            Cluster("SPEAKER_00", [1.0, 0.0], 10),
            Cluster("SPEAKER_01", [0.0, 1.0], 2),
        ])
        assert doc.save() is True
        original_label = "SPEAKER_00"
        other_label = "SPEAKER_01"
        apply = doc.apply_manual
    name = "A Person With Five Words"
    expected_body = doc.body.replace(f"] {original_label}:", f"] {name}:")
    assert apply({"SPEAKER_00": name}) == []
    assert doc.save() is True
    _persisted(doc, expected_body, [name, other_label])
    reloaded = load_identity(doc.path)
    assert reloaded.body == expected_body
    assert reloaded.frontmatter["speakers"] == [name, other_label]
    assert (reloaded.sidecar.clusters[0].name, reloaded.sidecar.clusters[0].source) == (name, "manual")
    assert set(build_voiceprints(tmp_path)) == {name}
    assert "SPEAKER_00" not in {target.key for target in reloaded.manual_targets}
    assert "SPEAKER_00" not in {target.key for target in reloaded.audit_targets}


@pytest.mark.parametrize("workflow", ["manual", "audit"])
@pytest.mark.parametrize("answer", ["Invalid: Name", "Invalid\nName"])
def test_non_roundtrippable_name_is_explicitly_rejected_without_writes(tmp_path, workflow, answer):
    if workflow == "audit":
        doc = _automatic(tmp_path)
        apply = doc.apply_audit
    else:
        doc = new_identity(tmp_path / "target.txt", dict(METADATA), RAW_BODY, [
            Cluster("SPEAKER_00", [1.0, 0.0], 10),
            Cluster("SPEAKER_01", [0.0, 1.0], 2),
        ])
        assert doc.save() is True
        apply = doc.apply_manual
    before = _snapshot(doc.path)
    original_body = doc.body
    original_cluster = asdict(doc.sidecar.clusters[0])
    _assert_rejected(apply({"SPEAKER_00": answer}), ["SPEAKER_00"])
    assert doc.body == original_body
    assert asdict(doc.sidecar.clusters[0]) == original_cluster
    assert doc.save() is False
    assert _snapshot(doc.path) == before


@pytest.mark.parametrize("workflow", ["manual", "audit"])
@pytest.mark.parametrize("answer", ["SPEAKER_00", "SPEAKER_99"])
def test_anonymous_placeholder_answer_skips_without_enrolling_or_writing(tmp_path, workflow, answer):
    if workflow == "audit":
        doc = _automatic(tmp_path)
        apply = doc.apply_audit
    else:
        doc = new_identity(tmp_path / "target.txt", dict(METADATA), RAW_BODY, [
            Cluster("SPEAKER_00", [1.0, 0.0], 10),
            Cluster("SPEAKER_01", [0.0, 1.0], 2),
        ])
        assert doc.save() is True
        apply = doc.apply_manual
    before = _snapshot(doc.path)
    original_body = doc.body
    original_cluster = asdict(doc.sidecar.clusters[0])
    assert apply({"SPEAKER_00": answer}) == []
    assert doc.save() is False
    assert _snapshot(doc.path) == before
    assert doc.body == original_body
    assert asdict(doc.sidecar.clusters[0]) == original_cluster
    assert build_voiceprints(tmp_path) == {}
