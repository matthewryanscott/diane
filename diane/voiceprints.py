"""Speaker voiceprints: name speaker clusters from earlier, named transcripts.

Each transcript can have a sidecar `<stem>.voice.json` that records, per speaker
cluster, its CAM++ centroid, speech time, and name with the name's source:

- manual: a name entered in `diane speakers`
- confirmed: an auto name accepted in `diane speakers audit`
- auto: applied at transcription time because the centroid matched a voiceprint

A person's voiceprint is the mean of the L2-normalized centroids of their manual
and confirmed clusters, derived on the fly, so fixing a name updates it. Auto
names never feed voiceprints. Matching uses the fixed MATCH_THRESHOLD below;
it is an operating setting, not a guarantee of identity accuracy.

IdentityDocument owns matching, pending targets, and staged identity decisions.
Successful saves derive speakers frontmatter from the final body and synchronize
only accepted cluster decisions. Complete optional provenance binds original
labels to their rendered labels and ordered occurrences, not text or timestamps;
legacy or invalid evidence never proves a named label safe to correct.

Loading and applying answers do not write. A new document or an accepted mutation
is saved transcript first, then sidecar. These writes are not a transaction:
PartialIdentityWriteError means the transcript was written but its sidecar was not.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from diane.db import Recording
from diane.frontmatter import build_output, frontmatter_from_recording, parse_frontmatter
from diane.speakers import (
    extract_speakers,
    replace_speakers,
    speaker_labels,
    unnamed_speakers,
)

SIDECAR_SUFFIX = ".voice.json"
SIDECAR_VERSION = 1
MATCH_THRESHOLD = 0.86
ENROLLING_SOURCES = frozenset({"manual", "confirmed"})
PROVENANCE_VERSION = 1


@dataclass
class Cluster:
    speaker: str
    centroid: list[float]
    speech_s: float
    name: str | None = None
    source: str | None = None
    score: float | None = None


@dataclass
class Sidecar:
    clusters: list[Cluster] = field(default_factory=list)
    embedding_model: str = "senko CAM++"
    diarizer: str = "nemotron"
    origin: str = "transcription"
    provenance: dict | None = None


def sidecar_path(transcript_path: Path) -> Path:
    return transcript_path.with_suffix(SIDECAR_SUFFIX)


def read_sidecar(transcript_path: Path) -> Sidecar | None:
    path = sidecar_path(transcript_path)
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    data = json.loads(text)
    return Sidecar(
        clusters=[Cluster(**c) for c in data.get("clusters", [])],
        embedding_model=data.get("embedding_model", "senko CAM++"),
        diarizer=data.get("diarizer", "nemotron"),
        origin=data.get("origin", "transcription"),
        provenance=data.get("provenance"),
    )


def write_sidecar(transcript_path: Path, sidecar: Sidecar) -> None:
    data = {"version": SIDECAR_VERSION, **asdict(sidecar)}
    sidecar_path(transcript_path).write_text(json.dumps(data, indent=1) + "\n")


def _normalized(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(_normalized(a), _normalized(b), strict=True))


def build_voiceprints(
    transcripts_dir: Path, *, exclude: Path | None = None
) -> dict[str, list[float]]:
    """Mean normalized centroid from named manual/confirmed clusters, never placeholders."""
    sums: dict[str, list[float]] = {}
    for path in sorted(transcripts_dir.glob(f"*{SIDECAR_SUFFIX}")):
        transcript = path.with_name(path.name.removesuffix(SIDECAR_SUFFIX) + ".txt")
        if exclude is not None and transcript == exclude:
            continue
        sidecar = read_sidecar(transcript)
        for cluster in sidecar.clusters if sidecar else []:
            if (
                cluster.name
                and cluster.source in ENROLLING_SOURCES
                and not unnamed_speakers(f"[00:00] {cluster.name}: identity")
            ):
                unit = _normalized(cluster.centroid)
                acc = sums.setdefault(cluster.name, [0.0] * len(unit))
                sums[cluster.name] = [s + x for s, x in zip(acc, unit, strict=True)]
    return {name: _normalized(total) for name, total in sums.items()}


def match(
    clusters: list[Cluster],
    voiceprints: dict[str, list[float]],
    *,
    threshold: float = MATCH_THRESHOLD,
) -> dict[str, tuple[str, float]]:
    """Map cluster speaker label → (name, score) for clusters at or above threshold.

    Two clusters may get the same name: the diarizer sometimes splits one person.
    """
    matches: dict[str, tuple[str, float]] = {}
    for cluster in clusters:
        scored = [(cosine(cluster.centroid, vp), name) for name, vp in voiceprints.items()]
        if scored:
            score, name = max(scored)
            if score >= threshold:
                matches[cluster.speaker] = (name, round(score, 4))
    return matches


@dataclass(frozen=True)
class IdentityTarget:
    """A pending decision keyed by the immutable, original speaker label."""

    key: str
    label: str
    score: float | None = None
    speech_s: float = 0.0
    shared: bool = False


@dataclass(frozen=True)
class IdentityRejection:
    key: str
    label: str
    reason: str


class PartialIdentityWriteError(RuntimeError):
    """The transcript was saved, but the subsequent sidecar write failed."""

    def __init__(self, transcript_path: Path):
        self.transcript_path = transcript_path
        self.sidecar_path = sidecar_path(transcript_path)
        super().__init__(
            f"Transcript written to {transcript_path}, but identity sidecar "
            f"{self.sidecar_path} could not be written; identity save is incomplete."
        )


def _provenance_fingerprint(labels: dict[str, str], body: str) -> str:
    payload = {"labels": labels, "occurrences": speaker_labels(body)}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _provenance(labels: dict[str, str], body: str) -> dict:
    return {
        "version": PROVENANCE_VERSION,
        "labels": labels,
        "fingerprint": _provenance_fingerprint(labels, body),
    }


def _validated_labels(sidecar: Sidecar | None, body: str) -> dict[str, str] | None:
    """Return complete evidence, or unknown; never reconstruct legacy attribution."""
    evidence = sidecar.provenance if sidecar is not None else None
    if not isinstance(evidence, dict) or evidence.get("version") != PROVENANCE_VERSION:
        return None
    labels = evidence.get("labels")
    if not isinstance(labels, dict) or not all(
        isinstance(key, str) and key and isinstance(label, str) and label
        for key, label in labels.items()
    ):
        return None
    if evidence.get("fingerprint") != _provenance_fingerprint(labels, body):
        return None
    if not set(extract_speakers(body)).issubset(labels.values()):
        return None
    if any(
        labels.get(cluster.speaker) != (cluster.name or cluster.speaker)
        for cluster in sidecar.clusters
    ):
        return None
    return labels


def _label_groups(
    labels: dict[str, str] | None, sidecar: Sidecar | None
) -> dict[str, list[str]]:
    """Group original keys; sidecar-only groups can show sharing, not uniqueness."""
    groups: dict[str, list[str]] = {}
    if labels is not None:
        pairs = labels.items()
    elif sidecar is not None:
        pairs = (
            (cluster.speaker, cluster.name or cluster.speaker)
            for cluster in sidecar.clusters
        )
    else:
        pairs = ()
    for key, label in pairs:
        groups.setdefault(label, []).append(key)
    return groups


_AMBIGUOUS_IDENTITY = (
    "Speaker attribution is unavailable or ambiguous; nothing for this correction "
    "changed. Independent per-cluster correction cannot safely distinguish this "
    "display label. Preserve the existing transcript and sidecar privately and "
    "produce a fresh transcription with provenance before correcting it."
)


@dataclass
class IdentityDocument:
    """In-memory identity state; only new or accepted mutations cause save writes.

    Provenance is optional sidecar metadata of the form
    ``{version: 1, labels: {original: rendered}, fingerprint: sha256}``.
    The digest covers the entire map plus ordered rendered label occurrences.
    Invalid evidence remains unknown, including after safe legacy manual naming.
    """

    path: Path
    frontmatter: dict
    body: str
    sidecar: Sidecar | None = None
    matches: dict[str, tuple[str, float]] = field(default_factory=dict)
    _dirty: bool = field(default=False, repr=False)

    @property
    def manual_targets(self) -> list[IdentityTarget]:
        labels = _validated_labels(self.sidecar, self.body)
        groups = _label_groups(labels, self.sidecar)
        clusters = {
            cluster.speaker: cluster
            for cluster in self.sidecar.clusters
        } if self.sidecar is not None else {}
        targets = []
        for label in unnamed_speakers(self.body):
            keys = groups[label] if labels is not None else [label]
            for key in keys:
                cluster = clusters.get(key)
                targets.append(IdentityTarget(
                    key=key,
                    label=label,
                    score=cluster.score if cluster is not None else None,
                    speech_s=float(cluster.speech_s) if cluster is not None else 0.0,
                    shared=len(groups.get(label, [])) > 1,
                ))
        return targets

    @property
    def audit_targets(self) -> list[IdentityTarget]:
        if self.sidecar is None:
            return []
        labels = _validated_labels(self.sidecar, self.body)
        groups = _label_groups(labels, self.sidecar)
        return [
            IdentityTarget(
                key=cluster.speaker,
                label=cluster.name,
                score=cluster.score,
                speech_s=float(cluster.speech_s),
                shared=len(groups.get(cluster.name, [])) > 1,
            )
            for cluster in self.sidecar.clusters
            if cluster.source == "auto" and cluster.name
        ]

    def apply_manual(self, answers: dict[str, str]) -> list[IdentityRejection]:
        """Name pending anonymous labels; blank or placeholder answers remain pending."""
        return self._apply(answers, audit=False)

    def apply_audit(self, answers: dict[str, str]) -> list[IdentityRejection]:
        """Confirm explicit auto targets, or safely correct unique proposals."""
        return self._apply(answers, audit=True)

    def _apply(
        self, answers: dict[str, str], *, audit: bool
    ) -> list[IdentityRejection]:
        # Freeze evidence, targets, and groups before considering any answer.
        labels = _validated_labels(self.sidecar, self.body)
        groups = _label_groups(labels, self.sidecar)
        clusters = {
            cluster.speaker: cluster
            for cluster in self.sidecar.clusters
        } if self.sidecar is not None else {}
        targets = self.audit_targets if audit else self.manual_targets
        by_key = {target.key: target for target in targets}
        proposals = {
            key: answer.strip()
            for key, answer in answers.items()
            if key in by_key
        }
        existing_labels = set(extract_speakers(self.body)) | set(groups)
        rejections = [
            IdentityRejection(key, key, "There is no pending identity target for this key.")
            for key in answers if key not in by_key
        ]
        accepted: list[tuple[IdentityTarget, str, bool]] = []
        for target in targets:
            if target.key not in proposals:
                continue
            answer = proposals[target.key]
            if not audit and not answer:
                continue
            confirmation = audit and (not answer or answer == target.label)
            if confirmation:
                accepted.append((target, target.label, True))
                continue
            # An anonymous diarizer label is not a person name or enrollment.
            if unnamed_speakers(f"[00:00] {answer}: identity"):
                continue
            if speaker_labels(f"[00:00] {answer}: identity") != [answer]:
                rejections.append(IdentityRejection(
                    target.key,
                    target.label,
                    "This name cannot be represented as one transcript speaker label; "
                    "nothing for this correction changed.",
                ))
                continue
            if labels is not None:
                safe = groups.get(target.label) == [target.key]
            else:
                # Only a still-raw anonymous label can be replaced without proof.
                # Existing and same-pass merges would lose attribution forever.
                cluster = clusters.get(target.key)
                same_pass_merge = any(
                    key != target.key and other == answer
                    for key, other in proposals.items() if other
                )
                safe = (
                    not audit
                    and target.key == target.label
                    and (
                        cluster is None
                        or (cluster.name or cluster.speaker) == target.label
                    )
                    and groups.get(target.label, []) in ([], [target.key])
                    and (answer == target.label or answer not in existing_labels)
                    and not same_pass_merge
                )
            if not safe:
                rejections.append(IdentityRejection(
                    target.key, target.label, _AMBIGUOUS_IDENTITY
                ))
                continue
            accepted.append((target, answer, False))

        replacements = {
            target.label: name
            for target, name, confirmation in accepted
            if not confirmation and name != target.label
        }
        new_body = replace_speakers(self.body, replacements)
        changed = new_body != self.body
        for target, name, confirmation in accepted:
            cluster = clusters.get(target.key)
            if cluster is None:
                continue
            source = "confirmed" if confirmation else "manual"
            score = cluster.score if confirmation else None
            if (cluster.name, cluster.source, cluster.score) != (name, source, score):
                cluster.name, cluster.source, cluster.score = name, source, score
                changed = True
        if changed:
            self.body = new_body
            self.frontmatter["speakers"] = extract_speakers(self.body)
            if labels is not None:
                updated = dict(labels)
                for target, name, _ in accepted:
                    updated[target.key] = name
                self.sidecar.provenance = _provenance(updated, self.body)
            self._dirty = True
        return rejections

    def save(self) -> bool:
        """Write staged changes transcript first; report a sidecar failure explicitly."""
        if not self._dirty:
            return False
        self.frontmatter["speakers"] = extract_speakers(self.body)
        self.path.write_text(build_output(self.frontmatter, self.body))
        if self.sidecar is not None:
            try:
                write_sidecar(self.path, self.sidecar)
            except Exception as exc:
                raise PartialIdentityWriteError(self.path) from exc
        self._dirty = False
        return True


def new_identity(
    path: Path,
    frontmatter: dict,
    raw_body: str,
    clusters: list[Cluster],
    *,
    transcripts_dir: Path | None = None,
) -> IdentityDocument:
    """Capture complete raw provenance, then apply unconfirmed enrollment matches."""
    owned_clusters = [replace(cluster) for cluster in clusters]
    labels = {label: label for label in extract_speakers(raw_body)}
    for cluster in owned_clusters:
        labels.setdefault(cluster.speaker, cluster.speaker)
    matches = match(
        owned_clusters,
        build_voiceprints(
            transcripts_dir if transcripts_dir is not None else path.parent,
            exclude=path,
        ),
    )
    for cluster in owned_clusters:
        if cluster.speaker in matches:
            name, score = matches[cluster.speaker]
            cluster.name, cluster.source, cluster.score = name, "auto", score
            labels[cluster.speaker] = name
    body = replace_speakers(raw_body, {
        key: label for key, label in labels.items() if key != label
    })
    fm = dict(frontmatter)
    fm["speakers"] = extract_speakers(body)
    sidecar = Sidecar(
        clusters=owned_clusters,
        provenance=_provenance(labels, body),
    )
    return IdentityDocument(path, fm, body, sidecar, matches, _dirty=True)


def load_identity(path: Path, recording: Recording | None = None) -> IdentityDocument:
    """Load without writing or normalizing; missing sidecars stay transcript-only."""
    fm, body = parse_frontmatter(path.read_text())
    if fm is None:
        fm = frontmatter_from_recording(recording) if recording is not None else {}
    return IdentityDocument(path, fm, body, read_sidecar(path))
