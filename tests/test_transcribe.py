"""Product transcription through Scribe, replacing only external model adapters."""

import threading
import wave
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import scribe

from diane.db import Recording
from diane.transcribe import (
    TranscriptionResult,
    format_diarized_transcript,
    transcribe_recording,
)
from diane.voiceprints import Cluster


@pytest.fixture
def recording_audio(tmp_path):
    original = tmp_path / "20260405 1400 Planning.wav"
    with wave.open(str(original), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * 64000)
    return Recording(
        unique_id="0123456789abcdef",
        date=datetime(2026, 4, 5, 14),
        duration=timedelta(seconds=8),
        title="Planning",
        path=str(original),
    )


@pytest.fixture
def model_adapters(monkeypatch, recording_audio):
    """Keep preparation, scheduling, attribution and centroid computation real."""
    original = Path(recording_audio.path)
    original_bytes = original.read_bytes()
    states = []

    def install(failure=None):
        state = SimpleNamespace(
            started=threading.Event(),
            release=threading.Event(),
            exited=threading.Event(),
            diarized=False,
            prepared=set(),
        )
        states.append(state)

        def inspect_audio(path):
            path = Path(path)
            assert path != original
            assert path.is_file()
            with wave.open(str(path), "rb") as audio:
                assert audio.getframerate() == 16000
                assert audio.getnchannels() == 1
                assert audio.getsampwidth() == 2
                assert audio.getnframes() == 128000
            state.prepared.add(path)

        def recognize(path):
            inspect_audio(path)
            state.started.set()
            try:
                assert state.release.wait(5), "ASR did not overlap the dependent model work"
                assert Path(path).exists()
                if failure == "asr":
                    raise scribe.ScribeError("recognition unavailable")
                return [
                    scribe.TranscriptionSegment(
                        "Welcome to our weekly planning meeting. "
                        "Please review the revised project schedule. Thanks.",
                        0.0,
                        8.0,
                        (
                            scribe.Word("Welcome", 0.2, 0.3),
                            scribe.Word("to", 0.4, 0.5),
                            scribe.Word("our", 0.6, 0.7),
                            scribe.Word("weekly", 0.8, 0.9),
                            scribe.Word("planning", 1.0, 1.3),
                            scribe.Word("meeting.", 1.4, 1.8),
                            scribe.Word("Please", 3.2, 3.3),
                            scribe.Word("review", 3.4, 3.5),
                            scribe.Word("the", 3.6, 3.7),
                            scribe.Word("revised", 3.8, 3.9),
                            scribe.Word("project", 4.0, 4.3),
                            scribe.Word("schedule.", 4.4, 4.8),
                            scribe.Word("Thanks.", 7.2, 7.6),
                        ),
                    ),
                ]
            finally:
                assert Path(path).exists(), "prepared audio disappeared before ASR exited"
                state.exited.set()

        def diarize(path):
            inspect_audio(path)
            assert state.started.wait(5), "recognition was not launched before diarization"
            assert not state.exited.is_set()
            if failure == "diarization":
                state.release.set()
                raise scribe.ScribeError("diarizer unavailable")
            state.diarized = True
            return [
                scribe.DiarizationSegment("SPEAKER_00", 0.0, 2.3456),
                scribe.DiarizationSegment("SPEAKER_01", 3.0, 5.6789),
                scribe.DiarizationSegment("SPEAKER_02", 7.0, 7.8),
            ]

        class Embedder:
            def _generate_subsegments(self, regions, _accurate):
                assert state.diarized
                assert not state.exited.is_set()
                assert regions == [(0.0, 2.3456), (3.0, 5.6789), (7.0, 7.8)]
                # No usable model window for SPEAKER_02: keep its speech, not a fake centroid.
                return [(0.0, 1.0), (1.0, 2.0), (3.0, 4.0), (4.0, 5.0)]

            def _extract_fbank_features(self, path, windows):
                inspect_audio(path)
                assert windows == [(0.0, 1.0), (1.0, 2.0), (3.0, 4.0), (4.0, 5.0)]
                return None, None, [0, 1, 2, 3], 2

            def _generate_embeddings(self, *_args):
                assert state.diarized
                assert not state.exited.is_set()
                state.release.set()
                if failure == "embedding":
                    raise RuntimeError("voice model unavailable")
                return [[3.0, 4.0], [0.0, 5.0], [5.0, 0.0], [5.0, 0.0]]

        monkeypatch.setattr(scribe, "run_transcription", recognize)
        monkeypatch.setattr(scribe, "run_nemotron_diarization", diarize)
        monkeypatch.setattr(scribe, "_create_embedder", Embedder)
        return state

    yield install

    for state in states:
        assert state.exited.is_set()
        assert len(state.prepared) == 1
        assert all(not path.exists() for path in state.prepared)
    assert original.read_bytes() == original_bytes


def test_transcribe_recording_returns_attributed_product_output(recording_audio, model_adapters):
    state = model_adapters()
    phases = []

    result = transcribe_recording(recording_audio, on_step=lambda step, total: phases.append((step, total)))

    assert isinstance(result, TranscriptionResult)
    assert result.segments == [
        scribe.MergedSegment("SPEAKER_00", "Welcome to our weekly planning meeting.", 0.2, 1.8),
        scribe.MergedSegment("SPEAKER_01", "Please review the revised project schedule.", 3.2, 4.8),
        scribe.MergedSegment("SPEAKER_02", "Thanks.", 7.2, 7.6),
    ]
    assert len(result.clusters) == 2
    assert result.clusters[0].speaker == "SPEAKER_00"
    assert result.clusters[0].centroid == pytest.approx([0.3, 0.9])
    assert result.clusters[0].speech_s == 2.35
    assert result.clusters[1] == Cluster("SPEAKER_01", [1.0, 0.0], 2.68)
    assert result.clusters[0].name is None
    assert result.clusters[0].source is None
    assert phases == [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]
    assert state.exited.is_set()
    assert format_diarized_transcript(result.segments) == (
        "[00:00] SPEAKER_00: Welcome to our weekly planning meeting.\n\n"
        "[00:03] SPEAKER_01: Please review the revised project schedule.\n\n"
        "[00:07] SPEAKER_02: Thanks.\n"
    )


@pytest.mark.parametrize(
    ("failure", "message", "expected_phases"),
    [
        ("diarization", "diarizer unavailable", [(0, 5), (1, 5), (2, 5)]),
        ("embedding", "voice model unavailable", [(0, 5), (1, 5), (2, 5), (3, 5)]),
        ("asr", "recognition unavailable", [(0, 5), (1, 5), (2, 5), (3, 5)]),
    ],
)
def test_transcribe_recording_failure_returns_no_result(
    recording_audio, model_adapters, failure, message, expected_phases,
):
    state = model_adapters(failure)
    phases = []
    results = []

    with pytest.raises(scribe.ScribeError, match=message):
        results.append(
            transcribe_recording(recording_audio, on_step=lambda step, total: phases.append((step, total)))
        )

    assert results == []
    assert phases == expected_phases
    assert state.exited.is_set()
    assert not any(path.exists() for path in state.prepared)


@pytest.mark.parametrize(
    ("segments", "expected"),
    [
        ([], ""),
        (
            [
                scribe.MergedSegment("A", "First.", 59.9, 60.9),
                scribe.MergedSegment("A", "After a pause.", 63.0, 64.0),
                scribe.MergedSegment("A", "Unfinished", 65.0, 66.0),
                scribe.MergedSegment("A", "thought.", 70.0, 71.0),
                scribe.MergedSegment("B", "An hour later.", 3601.9, 3603.0),
            ],
            "[00:59] A: First.\n\n"
            "[01:03] A: After a pause. Unfinished thought.\n\n"
            "[01:00:01] B: An hour later.\n",
        ),
        (
            [
                scribe.MergedSegment("A", "A long turn.", 0.0, 120.0),
                scribe.MergedSegment("A", "Next sentence.", 120.1, 121.0),
            ],
            "[00:00] A: A long turn.\n\n[02:00] A: Next sentence.\n",
        ),
    ],
)
def test_diane_formats_relative_timestamps_and_speaker_turns(segments, expected):
    assert format_diarized_transcript(segments) == expected
