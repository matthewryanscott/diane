"""Tests for the scribe module."""

import json
import shutil
import subprocess
import sys
import threading
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scribe import (
    UNKNOWN_SPEAKER,
    DiarizationSegment,
    MergedSegment,
    ScribeError,
    TranscriptionSegment,
    Word,
    _create_diarizer,
    _find_ffmpeg,
    _find_uvx,
    _label_by_first_appearance,
    _prepare_audio,
    diarize,
    format_json,
    format_text,
    format_transcript_json,
    format_transcript_text,
    main,
    merge,
    run_diarization,
    run_nemotron_diarization,
    run_transcription,
    transcribe_audio,
)


class TestMerge:
    """Tests for merge()."""

    def test_perfect_overlap(self):
        diarization = [DiarizationSegment("A", 0.0, 5.0)]
        transcription = [TranscriptionSegment("hello", 0.0, 5.0)]
        result = merge(diarization, transcription)
        assert result == [MergedSegment("A", "hello", 0.0, 5.0)]

    def test_partial_overlap_picks_best(self):
        diarization = [
            DiarizationSegment("A", 0.0, 3.0),
            DiarizationSegment("B", 2.0, 6.0),
        ]
        transcription = [TranscriptionSegment("hello", 1.0, 5.0)]
        result = merge(diarization, transcription)
        # B overlaps 2.0-5.0 = 3.0s, A overlaps 1.0-3.0 = 2.0s
        assert result[0].speaker == "B"

    def test_no_overlap_gives_unknown(self):
        diarization = [DiarizationSegment("A", 10.0, 15.0)]
        transcription = [TranscriptionSegment("hello", 0.0, 5.0)]
        result = merge(diarization, transcription)
        assert result[0].speaker == UNKNOWN_SPEAKER

    def test_empty_diarization(self):
        transcription = [TranscriptionSegment("hello", 0.0, 5.0)]
        result = merge([], transcription)
        assert result[0].speaker == UNKNOWN_SPEAKER

    def test_empty_transcription(self):
        diarization = [DiarizationSegment("A", 0.0, 5.0)]
        result = merge(diarization, [])
        assert result == []

    def test_both_empty(self):
        assert merge([], []) == []

    def test_multiple_speakers(self):
        diarization = [
            DiarizationSegment("A", 0.0, 5.0),
            DiarizationSegment("B", 5.0, 10.0),
        ]
        transcription = [
            TranscriptionSegment("first", 0.0, 4.0),
            TranscriptionSegment("second", 6.0, 9.0),
        ]
        result = merge(diarization, transcription)
        assert result[0].speaker == "A"
        assert result[1].speaker == "B"

    def test_preserves_timestamps(self):
        diarization = [DiarizationSegment("A", 0.0, 10.0)]
        transcription = [TranscriptionSegment("hi", 1.5, 3.7)]
        result = merge(diarization, transcription)
        assert result[0].start == 1.5
        assert result[0].end == 3.7

    def test_splits_segment_at_word_level_speaker_change(self):
        diarization = [
            DiarizationSegment("A", 0.0, 2.0),
            DiarizationSegment("B", 2.0, 5.0),
        ]
        words = (
            Word("So", 0.0, 0.5),
            Word("I", 0.6, 1.0),
            Word("think", 1.0, 1.8),
            Word("that", 2.1, 2.4),
            Word("we", 2.5, 2.8),
            Word("should", 2.9, 3.4),
            Word("ship.", 3.5, 4.0),
        )
        transcription = [TranscriptionSegment("So I think that we should ship.", 0.0, 4.0, words)]
        result = merge(diarization, transcription)
        assert result == [
            MergedSegment("A", "So I think", 0.0, 1.8),
            MergedSegment("B", "that we should ship.", 2.1, 4.0),
        ]

    def test_short_island_takes_surrounding_speaker(self):
        diarization = [
            DiarizationSegment("A", 0.0, 2.0),
            DiarizationSegment("B", 2.0, 3.0),
            DiarizationSegment("A", 3.0, 6.0),
        ]
        words = (
            Word("So", 0.0, 0.5),
            Word("I", 0.6, 1.0),
            Word("think,", 1.0, 1.8),
            Word("yeah", 2.1, 2.6),
            Word("we", 3.1, 3.4),
            Word("ship.", 3.5, 4.0),
        )
        transcription = [TranscriptionSegment("So I think, yeah we ship.", 0.0, 4.0, words)]
        result = merge(diarization, transcription)
        assert result == [MergedSegment("A", "So I think, yeah we ship.", 0.0, 4.0)]

    def test_short_edge_run_joins_neighbour_without_pause(self):
        diarization = [
            DiarizationSegment("A", 0.0, 1.0),
            DiarizationSegment("B", 1.0, 5.0),
        ]
        words = (
            Word("Well", 0.5, 0.95),
            Word("that", 1.0, 1.3),
            Word("is", 1.35, 1.5),
            Word("the", 1.55, 1.7),
            Word("plan.", 1.75, 2.2),
        )
        transcription = [TranscriptionSegment("Well that is the plan.", 0.5, 2.2, words)]
        result = merge(diarization, transcription)
        assert result == [MergedSegment("B", "Well that is the plan.", 0.5, 2.2)]

    def test_short_edge_run_after_pause_keeps_its_speaker(self):
        diarization = [
            DiarizationSegment("A", 0.0, 1.0),
            DiarizationSegment("B", 1.2, 5.0),
        ]
        words = (
            Word("Okay.", 0.3, 0.8),
            Word("So", 1.3, 1.5),
            Word("the", 1.55, 1.7),
            Word("plan", 1.75, 2.0),
            Word("works.", 2.05, 2.4),
        )
        transcription = [TranscriptionSegment("Okay. So the plan works.", 0.3, 2.4, words)]
        result = merge(diarization, transcription)
        assert [s.speaker for s in result] == ["A", "B"]

    def test_word_in_gap_takes_nearest_speaker(self):
        diarization = [
            DiarizationSegment("A", 0.0, 1.0),
            DiarizationSegment("B", 5.0, 6.0),
        ]
        words = (Word("hello", 0.5, 1.0), Word("there", 1.2, 1.5))
        transcription = [TranscriptionSegment("hello there", 0.5, 1.5, words)]
        result = merge(diarization, transcription)
        assert result == [MergedSegment("A", "hello there", 0.5, 1.5)]

    def test_gap_words_keep_previous_word_speaker(self):
        # "how we" sits in a diarization gap closer to B, but has no overlap
        # evidence, so it stays with the sentence's current speaker.
        diarization = [
            DiarizationSegment("A", 0.0, 1.0),
            DiarizationSegment("B", 2.6, 5.0),
        ]
        words = (
            Word("and", 0.5, 1.0),
            Word("how", 2.0, 2.2),
            Word("we", 2.3, 2.5),
        )
        transcription = [TranscriptionSegment("and how we", 0.5, 2.5, words)]
        result = merge(diarization, transcription)
        assert result == [MergedSegment("A", "and how we", 0.5, 2.5)]

    def test_first_word_far_from_any_speaker_is_unknown(self):
        diarization = [DiarizationSegment("A", 0.0, 1.0)]
        words = (Word("later", 4.0, 4.5), Word("on", 4.6, 4.8))
        transcription = [TranscriptionSegment("later on", 4.0, 4.8, words)]
        result = merge(diarization, transcription)
        assert result == [MergedSegment(UNKNOWN_SPEAKER, "later on", 4.0, 4.8)]


class TestFormatText:
    """Tests for format_text()."""

    def test_basic(self):
        segments = [MergedSegment("A", "hello world", 0.0, 2.0)]
        assert format_text(segments) == "A: hello world\n"

    def test_consecutive_same_speaker_merged(self):
        segments = [
            MergedSegment("A", "hello", 0.0, 1.0),
            MergedSegment("A", "world", 1.0, 2.0),
        ]
        assert format_text(segments) == "A: hello world\n"

    def test_speaker_change(self):
        segments = [
            MergedSegment("A", "hello", 0.0, 1.0),
            MergedSegment("B", "world", 1.0, 2.0),
        ]
        result = format_text(segments)
        assert "A: hello" in result
        assert "B: world" in result

    def test_blank_line_between_speakers(self):
        segments = [
            MergedSegment("A", "hello", 0.0, 1.0),
            MergedSegment("B", "world", 1.0, 2.0),
        ]
        result = format_text(segments)
        assert "\n\n" in result

    def test_empty(self):
        assert format_text([]) == ""


class TestFormatJson:
    """Tests for format_json()."""

    def test_basic_structure(self):
        segments = [MergedSegment("A", "hello", 0.531, 2.874)]
        data = json.loads(format_json(segments))
        assert data["speakers"] == ["A"]
        assert len(data["segments"]) == 1
        assert data["segments"][0]["speaker"] == "A"
        assert data["segments"][0]["text"] == "hello"
        assert data["segments"][0]["start"] == 0.531
        assert data["segments"][0]["end"] == 2.874

    def test_empty(self):
        data = json.loads(format_json([]))
        assert data["speakers"] == []
        assert data["segments"] == []

    def test_speakers_sorted(self):
        segments = [
            MergedSegment("B", "first", 0.0, 1.0),
            MergedSegment("A", "second", 1.0, 2.0),
        ]
        data = json.loads(format_json(segments))
        assert data["speakers"] == ["A", "B"]

    def test_timestamps_rounded(self):
        segments = [MergedSegment("A", "hi", 1.23456789, 2.98765432)]
        data = json.loads(format_json(segments))
        assert data["segments"][0]["start"] == 1.235
        assert data["segments"][0]["end"] == 2.988


class TestFormatTranscriptText:
    """Tests for format_transcript_text()."""

    def test_basic(self):
        segments = [TranscriptionSegment("hello world", 0.0, 2.0)]
        assert format_transcript_text(segments) == "hello world\n"

    def test_multiple_segments_joined(self):
        segments = [
            TranscriptionSegment("hello", 0.0, 1.0),
            TranscriptionSegment("world", 1.0, 2.0),
        ]
        assert format_transcript_text(segments) == "hello world\n"

    def test_empty(self):
        assert format_transcript_text([]) == ""


class TestFormatTranscriptJson:
    """Tests for format_transcript_json()."""

    def test_basic_structure(self):
        segments = [TranscriptionSegment("hello", 0.531, 2.874)]
        data = json.loads(format_transcript_json(segments))
        assert "speakers" not in data
        assert len(data["segments"]) == 1
        assert "speaker" not in data["segments"][0]
        assert data["segments"][0]["text"] == "hello"
        assert data["segments"][0]["start"] == 0.531
        assert data["segments"][0]["end"] == 2.874

    def test_empty(self):
        data = json.loads(format_transcript_json([]))
        assert data["segments"] == []

    def test_timestamps_rounded(self):
        segments = [TranscriptionSegment("hi", 1.23456789, 2.98765432)]
        data = json.loads(format_transcript_json(segments))
        assert data["segments"][0]["start"] == 1.235
        assert data["segments"][0]["end"] == 2.988


class TestFindFfmpeg:
    """Tests for _find_ffmpeg()."""

    def test_finds_ffmpeg(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/ffmpeg")
        assert _find_ffmpeg() == "/usr/local/bin/ffmpeg"

    def test_missing_ffmpeg_raises(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        with pytest.raises(ScribeError, match="ffmpeg is required"):
            _find_ffmpeg()


class TestFindUvx:
    """Tests for _find_uvx()."""

    def test_finds_uvx(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/uvx")
        assert _find_uvx() == "/usr/local/bin/uvx"

    def test_missing_uvx_raises(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        with pytest.raises(ScribeError, match="uvx not found"):
            _find_uvx()


@pytest.fixture
def _mock_ffmpeg(monkeypatch):
    """Stub out ffmpeg discovery and subprocess.run for _prepare_audio tests."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_kw: subprocess.CompletedProcess([], 0),
    )


class TestPrepareAudio:
    """Tests for _prepare_audio()."""

    @pytest.mark.usefixtures("_mock_ffmpeg")
    def test_wav_still_converts(self, tmp_path):
        wav = tmp_path / "test.wav"
        wav.touch()
        with _prepare_audio(wav) as result:
            assert result.suffix == ".wav"
            assert result != wav

    @pytest.mark.usefixtures("_mock_ffmpeg")
    def test_non_wav_converts(self, tmp_path):
        mp4 = tmp_path / "test.mp4"
        mp4.touch()
        with _prepare_audio(mp4) as result:
            assert result.suffix == ".wav"
            assert result != mp4

    @pytest.mark.usefixtures("_mock_ffmpeg")
    def test_temp_file_cleaned_up(self, tmp_path):
        mp4 = tmp_path / "test.mp4"
        mp4.touch()
        temp_path = None
        with _prepare_audio(mp4) as result:
            temp_path = result
        assert not temp_path.exists()

    def test_ffmpeg_failure_raises(self, tmp_path, monkeypatch):
        mp4 = tmp_path / "test.mp4"
        mp4.touch()
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")

        def fail_run(*_args, **_kwargs):
            raise subprocess.CalledProcessError(1, "ffmpeg", stderr="bad codec")

        monkeypatch.setattr("subprocess.run", fail_run)
        with pytest.raises(ScribeError, match="ffmpeg conversion failed"), _prepare_audio(mp4):
            pass


class TestCreateDiarizer:
    """Tests for _create_diarizer()."""

    def test_missing_senko_raises(self, monkeypatch):
        import builtins  # noqa: PLC0415

        real_import = builtins.__import__

        def block_senko(name, *args, **kwargs):
            if name == "senko":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_senko)
        with pytest.raises(ScribeError, match="senko is not installed"):
            _create_diarizer()

    def test_init_failure_raises(self, monkeypatch):
        mock_senko = MagicMock()
        mock_senko.Diarizer.side_effect = RuntimeError("CoreML init failed")
        monkeypatch.setitem(sys.modules, "senko", mock_senko)
        with pytest.raises(ScribeError, match="Failed to initialize diarizer"):
            _create_diarizer()


class TestRunTranscription:
    """Tests for run_transcription()."""

    def _mock_parakeet(self, monkeypatch, json_data):
        """Set up mocks for run_transcription with given JSON output."""
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uvx")

        def fake_run(cmd, **_kwargs):
            idx = cmd.index("--output-dir")
            out_dir = cmd[idx + 1]
            out_file = Path(out_dir) / "output.json"
            out_file.write_text(json.dumps(json_data))
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("subprocess.run", fake_run)

    def test_segments_key(self, monkeypatch, tmp_path):
        data = {"segments": [{"text": "hello", "start": 0.0, "end": 1.0}]}
        self._mock_parakeet(monkeypatch, data)
        result = run_transcription(tmp_path / "audio.wav")
        assert len(result) == 1
        assert result[0].text == "hello"

    def test_sentences_key_fallback(self, monkeypatch, tmp_path):
        data = {"sentences": [{"text": "world", "start": 1.0, "end": 2.0}]}
        self._mock_parakeet(monkeypatch, data)
        result = run_transcription(tmp_path / "audio.wav")
        assert len(result) == 1
        assert result[0].text == "world"

    def test_uses_short_chunks(self, monkeypatch, tmp_path):
        calls = []
        self._mock_parakeet(monkeypatch, {"segments": []})
        real_run = subprocess.run

        def spy(cmd, **kwargs):
            calls.append(cmd)
            return real_run(cmd, **kwargs)

        monkeypatch.setattr("subprocess.run", spy)
        run_transcription(tmp_path / "audio.wav")
        cmd = calls[0]
        assert cmd[cmd.index("--chunk-duration") + 1] == "30"
        assert cmd[cmd.index("--overlap-duration") + 1] == "5"

    def test_tokens_grouped_into_words(self, monkeypatch, tmp_path):
        tokens = [
            {"text": " He", "start": 0.0, "end": 0.2},
            {"text": "y", "start": 0.2, "end": 0.3},
            {"text": ",", "start": 0.3, "end": 0.4},
            {"text": " Ale", "start": 0.5, "end": 0.7},
            {"text": "x", "start": 0.7, "end": 0.9},
            {"text": ".", "start": 0.9, "end": 1.0},
        ]
        data = {"sentences": [{"text": " Hey, Alex.", "start": 0.0, "end": 1.0, "tokens": tokens}]}
        self._mock_parakeet(monkeypatch, data)
        result = run_transcription(tmp_path / "audio.wav")
        # punctuation attaches without extending the word's end time
        assert result[0].words == (Word("Hey,", 0.0, 0.3), Word("Alex.", 0.5, 0.9))

    def test_spaced_punctuation_token_never_starts_a_word(self, monkeypatch, tmp_path):
        tokens = [
            {"text": " Yes", "start": 0.0, "end": 0.3},
            {"text": " .", "start": 0.3, "end": 0.5},
        ]
        data = {"sentences": [{"text": "Yes .", "start": 0.0, "end": 0.5, "tokens": tokens}]}
        self._mock_parakeet(monkeypatch, data)
        result = run_transcription(tmp_path / "audio.wav")
        assert result[0].words == (Word("Yes.", 0.0, 0.3),)

    def test_empty_output(self, monkeypatch, tmp_path):
        data = {"segments": []}
        self._mock_parakeet(monkeypatch, data)
        result = run_transcription(tmp_path / "audio.wav")
        assert result == []

    def test_missing_fields_skipped(self, monkeypatch, tmp_path):
        data = {"segments": [{"text": "ok"}, {"text": "hi", "start": 0.0, "end": 1.0}]}
        self._mock_parakeet(monkeypatch, data)
        result = run_transcription(tmp_path / "audio.wav")
        assert len(result) == 1
        assert result[0].text == "hi"

    def test_invalid_json_raises(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uvx")

        def fake_run(cmd, **_kwargs):
            idx = cmd.index("--output-dir")
            out_dir = cmd[idx + 1]
            out_file = Path(out_dir) / "output.json"
            out_file.write_text("{invalid")
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("subprocess.run", fake_run)
        with pytest.raises(ScribeError, match="invalid JSON"):
            run_transcription(Path("/fake/audio.wav"))

    def test_subprocess_failure_raises(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uvx")

        def fail_run(*_args, **_kwargs):
            raise subprocess.CalledProcessError(1, "parakeet-mlx", stderr="out of memory")

        monkeypatch.setattr("subprocess.run", fail_run)
        with pytest.raises(ScribeError, match="parakeet-mlx failed"):
            run_transcription(Path("/fake/audio.wav"))

    def test_no_json_output_raises(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uvx")
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, **_kw: subprocess.CompletedProcess(cmd, 0),
        )
        with pytest.raises(ScribeError, match="no JSON output"):
            run_transcription(Path("/fake/audio.wav"))


class TestRunDiarization:
    """Tests for run_diarization()."""

    def test_returns_segments(self):
        diarizer = MagicMock()
        diarizer.diarize.return_value = {
            "merged_segments": [
                {"speaker": "SPEAKER_01", "start": 0.0, "end": 2.5},
                {"speaker": "SPEAKER_02", "start": 2.5, "end": 5.0},
            ],
        }
        result = run_diarization(diarizer, Path("/fake/audio.wav"))
        assert len(result) == 2
        assert result[0] == DiarizationSegment("SPEAKER_01", 0.0, 2.5)
        assert result[1] == DiarizationSegment("SPEAKER_02", 2.5, 5.0)

    def test_none_result_returns_empty(self):
        diarizer = MagicMock()
        diarizer.diarize.return_value = None
        result = run_diarization(diarizer, Path("/fake/audio.wav"))
        assert result == []

    def test_empty_segments_returns_empty(self):
        diarizer = MagicMock()
        diarizer.diarize.return_value = {"merged_segments": []}
        result = run_diarization(diarizer, Path("/fake/audio.wav"))
        assert result == []

    def test_malformed_result_raises(self):
        diarizer = MagicMock()
        diarizer.diarize.return_value = {"wrong_key": []}
        with pytest.raises(ScribeError, match="Diarization failed"):
            run_diarization(diarizer, Path("/fake/audio.wav"))

    def test_wraps_diarizer_errors(self):
        diarizer = MagicMock()
        diarizer.diarize.side_effect = RuntimeError("boom")
        with pytest.raises(ScribeError, match="Diarization failed"):
            run_diarization(diarizer, Path("/fake/audio.wav"))


class TestNemotronDiarization:
    """Tests for run_nemotron_diarization() and diarize()."""

    def _mock_worker(self, monkeypatch, rows):
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uvx")
        calls = []

        def fake_run(cmd, **_kwargs):
            calls.append(cmd)
            out = Path(cmd[cmd.index("-c") + 3])
            out.write_text(json.dumps(rows))
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("subprocess.run", fake_run)
        return calls

    def test_segments_labeled_in_speaking_order(self, monkeypatch):
        self._mock_worker(monkeypatch, [[3, 5.0, 6.0], [1, 0.5, 2.0], [3, 2.5, 4.0]])
        result = run_nemotron_diarization(Path("/fake/audio.wav"))
        assert result == [
            DiarizationSegment("SPEAKER_00", 0.5, 2.0),
            DiarizationSegment("SPEAKER_01", 2.5, 4.0),
            DiarizationSegment("SPEAKER_01", 5.0, 6.0),
        ]

    def test_runs_in_isolated_uvx_env(self, monkeypatch):
        calls = self._mock_worker(monkeypatch, [])
        run_nemotron_diarization(Path("/fake/audio.wav"))
        cmd = calls[0]
        assert cmd[0] == "/usr/bin/uvx"
        assert cmd[cmd.index("--from") + 1] == "mlx-audio==0.5.6"
        assert "/fake/audio.wav" in cmd

    def test_empty_output(self, monkeypatch):
        self._mock_worker(monkeypatch, [])
        assert run_nemotron_diarization(Path("/fake/audio.wav")) == []

    def test_worker_failure_raises(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uvx")

        def fail(cmd, **_kwargs):
            raise subprocess.CalledProcessError(1, cmd, stderr="model load failed")

        monkeypatch.setattr("subprocess.run", fail)
        with pytest.raises(ScribeError, match="Nemotron diarization failed: model load failed"):
            run_nemotron_diarization(Path("/fake/audio.wav"))

    def test_missing_output_raises(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uvx")
        monkeypatch.setattr(
            "subprocess.run", lambda cmd, **_kw: subprocess.CompletedProcess(cmd, 0)
        )
        with pytest.raises(ScribeError, match="no usable output"):
            run_nemotron_diarization(Path("/fake/audio.wav"))

    def test_label_by_first_appearance_ties_break_on_index(self):
        result = _label_by_first_appearance([[2, 1.0, 2.0], [0, 1.0, 1.5]])
        assert [s.speaker for s in result] == ["SPEAKER_00", "SPEAKER_01"]
        assert result[0].end == 1.5

    def test_diarize_dispatches_to_senko(self, monkeypatch):
        diarizer = MagicMock()
        segment = {"speaker": "A", "start": 0.0, "end": 1.0}
        diarizer.diarize.return_value = {"merged_segments": [segment]}
        monkeypatch.setattr("scribe._create_diarizer", lambda: diarizer)
        result = diarize(Path("/fake/audio.wav"), diarizer="senko")
        assert result == [DiarizationSegment("A", 0.0, 1.0)]

    def test_diarize_unknown_raises(self):
        with pytest.raises(ScribeError, match="Unknown diarizer"):
            diarize(Path("/fake/audio.wav"), diarizer="nope")


class TestMain:
    """Tests for main() CLI entry point."""

    def test_file_not_found(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["scribe", "/nonexistent/audio.wav"])
        with pytest.raises(SystemExit, match="audio file not found"):
            main()

    def test_diarize_to_file(self, model_boundary, monkeypatch, capsys):
        out = model_boundary.audio.with_suffix(".txt")
        monkeypatch.setattr("sys.argv", ["scribe", str(model_boundary.audio), "-o", str(out)])
        main()
        assert out.read_text() == "SPEAKER_00: Hello\n\nSPEAKER_01: again.\n"
        diagnostics = capsys.readouterr().err
        assert "Found 2 speakers" in diagnostics
        assert "Transcribed 1 segments" in diagnostics
        assert "Merging results..." in diagnostics
        assert model_boundary.conversions == 1
        assert not model_boundary.prepared.exists()

    @pytest.mark.parametrize("output_format", ["text", "json"])
    def test_no_diarize_to_stdout(self, model_boundary, monkeypatch, capsys, output_format):
        def unexpected_diarization():
            pytest.fail("transcript-only CLI launched diarization")

        model_boundary.diarization_hook = unexpected_diarization
        monkeypatch.setattr(
            "sys.argv",
            [
                "scribe", str(model_boundary.audio), "-o", "-", "--no-diarize",
                "--format", output_format,
            ],
        )
        main()
        output = capsys.readouterr().out
        if output_format == "text":
            assert output == "Hello again.\n"
        else:
            assert json.loads(output) == {
                "segments": [{"text": "Hello again.", "start": 0.0, "end": 2.5}],
            }
        assert model_boundary.conversions == 1
        assert not model_boundary.prepared.exists()

    def test_default_output_path_json(self, model_boundary, monkeypatch):
        monkeypatch.setattr(
            "sys.argv", ["scribe", str(model_boundary.audio), "--format", "json"],
        )
        main()
        assert json.loads(model_boundary.audio.with_suffix(".json").read_text()) == {
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
            "segments": [
                {"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0, "end": 0.5},
                {"speaker": "SPEAKER_01", "text": "again.", "start": 2.0, "end": 2.5},
            ],
        }

    def test_selected_senko_diarizer(self, model_boundary, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["scribe", str(model_boundary.audio), "--diarizer", "senko", "-o", "-"],
        )
        main()
        assert model_boundary.senko_diarized

    def test_scribe_error_exits_without_output(self, model_boundary, monkeypatch):
        def fail():
            raise subprocess.CalledProcessError(1, "nemotron", stderr="diarization broke")

        model_boundary.diarization_hook = fail
        monkeypatch.setattr("sys.argv", ["scribe", str(model_boundary.audio)])
        with pytest.raises(SystemExit, match="diarization broke"):
            main()
        assert not model_boundary.audio.with_suffix(".txt").exists()
        assert model_boundary.asr_stopped.is_set()
        assert not model_boundary.prepared.exists()


def _assert_prepared_audio(path, state):
    assert Path(path) == state.prepared
    assert Path(path) != state.audio
    with wave.open(str(path), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        assert wav.getnframes() == 64000


def _run_boundary_recognition(cmd, state):
    _assert_prepared_audio(cmd[2], state)
    state.asr_started.set()
    try:
        state.asr_hook()
        out = Path(cmd[cmd.index("--output-dir") + 1]) / "recognized.json"
        out.write_text(json.dumps({
            "segments": [{
                "text": "Hello again.", "start": 0.0, "end": 2.5,
                "tokens": [
                    {"text": " Hello", "start": 0.0, "end": 0.5},
                    {"text": " again.", "start": 2.0, "end": 2.5},
                ],
            }],
        }))
        return subprocess.CompletedProcess(cmd, 0)
    finally:
        _assert_prepared_audio(cmd[2], state)
        state.asr_stopped.set()


@pytest.fixture
def model_boundary(tmp_path, monkeypatch):
    """Run real WAV normalization and Scribe logic; replace only model I/O."""
    real_which = shutil.which
    if real_which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for public audio-pipeline tests")
    real_run = subprocess.run
    audio = tmp_path / "meeting.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(b"\0" * 44100 * 4 * 4)
    state = SimpleNamespace(
        audio=audio,
        prepared=None,
        conversions=0,
        asr_started=threading.Event(),
        asr_stopped=threading.Event(),
        diarized=threading.Event(),
        asr_hook=lambda: None,
        diarization_hook=lambda: None,
        embedding_hook=lambda: None,
        senko_diarized=False,
        rows=[[5, 0.0, 1.0], [8, 2.0, 3.0]],
    )

    def model_run(cmd, **kwargs):
        if "--output-dir" in cmd:
            return _run_boundary_recognition(cmd, state)
        if "-c" in cmd:
            _assert_prepared_audio(cmd[cmd.index("-c") + 2], state)
            assert state.asr_started.wait(5), "ASR was not launched before diarization"
            state.diarization_hook()
            Path(cmd[cmd.index("-c") + 3]).write_text(json.dumps(state.rows))
            state.diarized.set()
            return subprocess.CompletedProcess(cmd, 0)
        state.conversions += 1
        state.prepared = Path(cmd[-2])
        return real_run(cmd, **kwargs)

    class ModelDiarizer:
        def __init__(self, **_kwargs):
            pass

        def diarize(self, path):
            state.senko_diarized = True
            _assert_prepared_audio(path, state)
            assert state.asr_started.wait(5)
            state.diarization_hook()
            state.diarized.set()
            return {"merged_segments": [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
                {"speaker": "SPEAKER_01", "start": 2.0, "end": 3.0},
            ]}

        def _generate_subsegments(self, regions, _accurate):
            assert state.diarized.is_set(), "embedding preceded diarization"
            return regions

        def _extract_fbank_features(self, path, subsegments):
            _assert_prepared_audio(path, state)
            assert subsegments == [(0.0, 1.0), (2.0, 3.0)]
            state.embedding_hook()
            return None, None, [0, 1], 2

        def _generate_embeddings(self, *_args):
            return [[3.0, 4.0], [0.0, 2.0]]

    monkeypatch.setattr(
        "shutil.which", lambda name: "/model-boundary/uvx" if name == "uvx" else real_which(name),
    )
    monkeypatch.setattr("subprocess.run", model_run)
    monkeypatch.setitem(sys.modules, "senko", SimpleNamespace(Diarizer=ModelDiarizer))
    return state


class _PipelineFailureObserver:
    """Observe public-operation failure while its real model worker stays blocked."""

    def __init__(self, audio, failure_stage, with_diagnostics):
        self.audio = audio
        self.with_diagnostics = with_diagnostics
        self.diagnostic_timeout = failure_stage == "diarization" and with_diagnostics
        self.expected_error = TimeoutError if failure_stage == "timeout" else ScribeError
        self.release_asr = threading.Event()
        self.failure_observed = threading.Event()
        self.finished = threading.Event()
        self.phases = []
        self.logs = []
        self.outcome = []

    def recognize(self):
        assert self.release_asr.wait(5), "test did not release the ASR worker"

    def log(self, message):
        self.logs.append(message)
        if self.diagnostic_timeout and message.startswith("Transcription also failed:"):
            self.failure_observed.set()

    def observe_failure(self, frame, event, arg):
        # Observe real exceptions at the public operation, without replacing
        # collection/unwinding or depending on private futures and line numbers.
        if frame.f_code is not transcribe_audio.__code__:
            return None
        if event == "exception" and isinstance(arg[1], self.expected_error):
            self.failure_observed.set()
        return self.observe_failure

    def invoke(self):
        previous_trace = sys.gettrace()
        if not self.diagnostic_timeout:
            sys.settrace(self.observe_failure)
        try:
            self.outcome.append(transcribe_audio(
                self.audio, include_embeddings=True,
                on_step=lambda index, total: self.phases.append((index, total)),
                on_log=self.log if self.with_diagnostics else None,
            ))
        except Exception as exc:  # noqa: BLE001
            self.outcome.append(exc)
        finally:
            if not self.diagnostic_timeout:
                sys.settrace(previous_trace)
            self.finished.set()


def _configure_pipeline_failure(state, monkeypatch, failure_stage, diagnostic_timeout):
    def fail_diarization():
        raise subprocess.CalledProcessError(1, "nemotron", stderr="diarization primary")

    def fail_embeddings():
        message = "embedding primary"
        raise RuntimeError(message)

    if failure_stage == "diarization":
        state.diarization_hook = fail_diarization
    elif failure_stage == "embedding":
        state.embedding_hook = fail_embeddings
    else:
        # Shorten the existing collection timeout, not the worker/shutdown policy.
        monkeypatch.setattr("scribe._FUTURE_TIMEOUT_S", 0)
    if diagnostic_timeout:
        monkeypatch.setattr("scribe._ERROR_FUTURE_TIMEOUT_S", 0)


class TestAudioPipeline:
    """Exercise the supported owner, not private preparation/executor wiring."""

    def test_overlaps_asr_with_diarization_and_embeddings(self, model_boundary, capsys):
        state = model_boundary
        release_asr = threading.Event()
        phases = []
        observations = []

        def recognize():
            assert release_asr.wait(5), "embeddings did not overlap the ASR worker"

        def diarize_model():
            assert not state.asr_stopped.is_set()
            observations.append("diarization")

        def embed_model():
            assert not state.asr_stopped.is_set()
            assert observations == ["diarization"]
            observations.append("embeddings")
            release_asr.set()

        state.asr_hook = recognize
        state.diarization_hook = diarize_model
        state.embedding_hook = embed_model

        def phase_started(index, total):
            phases.append((index, total))
            if index == 0:
                assert state.conversions == 0
            elif index == 1:
                assert state.prepared.exists()
                assert not state.asr_started.is_set()
            elif index == 2:
                assert not state.diarized.is_set()
            elif index == 3:
                assert observations == ["diarization"]
            elif index == 4:
                assert observations == ["diarization", "embeddings"]
                assert state.asr_stopped.is_set()

        result = transcribe_audio(
            state.audio, include_embeddings=True,
            on_step=phase_started,
        )
        assert phases == [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]
        assert observations == ["diarization", "embeddings"]
        assert result.segments == [
            MergedSegment("SPEAKER_00", "Hello", 0.0, 0.5),
            MergedSegment("SPEAKER_01", "again.", 2.0, 2.5),
        ]
        assert set(result.embeddings) == {"SPEAKER_00", "SPEAKER_01"}
        assert result.embeddings["SPEAKER_00"].centroid == pytest.approx([0.6, 0.8])
        assert result.embeddings["SPEAKER_01"].centroid == [0.0, 1.0]
        assert [(e.speech_s, e.n_subsegments) for e in result.embeddings.values()] == [
            (1.0, 1), (1.0, 1),
        ]
        assert state.conversions == 1
        assert state.asr_stopped.is_set()
        assert not state.prepared.exists()
        assert capsys.readouterr().err == ""

    def test_without_embeddings_skips_only_voiceprint_phase(self, model_boundary):
        phases = []

        def unexpected_embeddings():
            pytest.fail("optional embedding model was loaded")

        model_boundary.embedding_hook = unexpected_embeddings
        result = transcribe_audio(
            model_boundary.audio, on_step=lambda index, total: phases.append((index, total)),
        )
        assert result.embeddings == {}
        assert len(result.segments) == 2
        assert phases == [(0, 5), (1, 5), (2, 5), (4, 5)]

    @pytest.mark.parametrize("failure_stage", ["diarization", "embedding", "timeout"])
    @pytest.mark.parametrize("with_diagnostics", [False, True])
    def test_failure_waits_for_worker_before_cleanup(
        self, model_boundary, monkeypatch, failure_stage, with_diagnostics,
    ):
        state = model_boundary
        observer = _PipelineFailureObserver(state.audio, failure_stage, with_diagnostics)
        state.asr_hook = observer.recognize
        _configure_pipeline_failure(
            state, monkeypatch, failure_stage, observer.diagnostic_timeout,
        )
        worker = threading.Thread(target=observer.invoke)
        worker.start()
        try:
            assert observer.failure_observed.wait(5), (
                "pipeline did not observe the expected failure"
            )
            assert not observer.finished.is_set(), (
                "pipeline returned before its worker stopped"
            )
            assert state.prepared.exists()
            assert not state.asr_stopped.is_set()
        finally:
            observer.release_asr.set()
            worker.join(5)
        assert not worker.is_alive()
        assert observer.finished.is_set()
        assert state.asr_stopped.is_set()
        assert not state.prepared.exists()
        assert len(observer.outcome) == 1
        assert isinstance(observer.outcome[0], observer.expected_error)
        if failure_stage != "timeout":
            assert f"{failure_stage} primary" in str(observer.outcome[0])
        expected = [(0, 5), (1, 5), (2, 5)]
        if failure_stage != "diarization":
            expected.append((3, 5))
        assert observer.phases == expected
        if observer.diagnostic_timeout:
            assert any(
                message.startswith("Transcription also failed:") for message in observer.logs
            )
        assert not any("Merging" in message for message in observer.logs)

    @pytest.mark.parametrize("with_diagnostics", [False, True])
    def test_diarization_keeps_primary_and_optional_secondary_diagnostic(
        self, model_boundary, with_diagnostics,
    ):
        def fail_asr():
            raise subprocess.CalledProcessError(1, "parakeet", stderr="recognition secondary")

        def fail_diarization():
            raise subprocess.CalledProcessError(1, "nemotron", stderr="diarization primary")

        model_boundary.asr_hook = fail_asr
        model_boundary.diarization_hook = fail_diarization
        logs = []
        with pytest.raises(ScribeError, match="Nemotron diarization failed: diarization primary"):
            transcribe_audio(
                model_boundary.audio, on_log=logs.append if with_diagnostics else None,
            )
        secondary = [message for message in logs if "Transcription also failed:" in message]
        assert bool(secondary) == with_diagnostics
        if with_diagnostics:
            assert "recognition secondary" in secondary[0]
        assert not model_boundary.prepared.exists()

    def test_embedding_failure_has_no_secondary_asr_diagnostic(self, model_boundary):
        def fail_asr():
            raise subprocess.CalledProcessError(1, "parakeet", stderr="recognition secondary")

        def fail_embeddings():
            message = "embedding primary"
            raise RuntimeError(message)

        model_boundary.asr_hook = fail_asr
        model_boundary.embedding_hook = fail_embeddings
        logs = []
        with pytest.raises(ScribeError, match="embedding primary"):
            transcribe_audio(model_boundary.audio, include_embeddings=True, on_log=logs.append)
        assert not any("Transcription also failed:" in message for message in logs)
        assert model_boundary.asr_stopped.is_set()
        assert not model_boundary.prepared.exists()

    def test_asr_failure_is_collected_after_embeddings(self, model_boundary):
        phases = []
        embedded = []

        def fail_asr():
            raise subprocess.CalledProcessError(1, "parakeet", stderr="recognition primary")

        model_boundary.asr_hook = fail_asr
        model_boundary.embedding_hook = lambda: embedded.append(True)
        with pytest.raises(ScribeError, match="parakeet-mlx failed: recognition primary"):
            transcribe_audio(
                model_boundary.audio, include_embeddings=True,
                on_step=lambda index, total: phases.append((index, total)),
            )
        assert embedded == [True]
        assert phases == [(0, 5), (1, 5), (2, 5), (3, 5)]
        assert not model_boundary.prepared.exists()

    def test_real_conversion_failure_stops_before_models(self, model_boundary):
        model_boundary.audio.write_bytes(b"not an audio stream")
        phases = []
        with pytest.raises(ScribeError, match="ffmpeg conversion failed"):
            transcribe_audio(
                model_boundary.audio,
                on_step=lambda index, total: phases.append((index, total)),
            )
        assert phases == [(0, 5)]
        assert model_boundary.conversions == 1
        assert not model_boundary.asr_started.is_set()
        assert not model_boundary.prepared.exists()

    def test_missing_ffmpeg_stops_in_preparing_phase(self, tmp_path, monkeypatch):
        phases = []
        monkeypatch.setattr("shutil.which", lambda _name: None)
        with pytest.raises(ScribeError, match="ffmpeg is required"):
            transcribe_audio(
                tmp_path / "original.wav",
                on_step=lambda index, total: phases.append((index, total)),
            )
        assert phases == [(0, 5)]

    def test_no_speech_warns_and_retains_unknown_words(self, model_boundary):
        model_boundary.rows = []
        logs = []
        result = transcribe_audio(model_boundary.audio, include_embeddings=True, on_log=logs.append)
        assert result.segments == [MergedSegment(UNKNOWN_SPEAKER, "Hello again.", 0.0, 2.5)]
        assert result.embeddings == {}
        assert "Warning: no speech segments detected" in logs
        assert "Warning: 1 segments could not be attributed to a speaker" in logs
