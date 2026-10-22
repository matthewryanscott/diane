# scribe

Local transcription and speaker diarization with
[parakeet-mlx](https://github.com/senstella/parakeet-mlx) and
[Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
(through its [mlx-audio](https://github.com/Blaizzy/mlx-audio) port), with
[senko](https://github.com/narcotic-sh/senko) as an alternative diarizer.

Scribe produces timestamped, speaker-attributed transcripts for recurring meetings with 2–6 people.
It accepts audio or video formats that ffmpeg can decode and converts them to 16 kHz mono WAV.
It does not extract structured meeting data or generate summaries.

Transcription and diarization run in parallel on Apple Silicon.
Parakeet and the default Nemotron diarizer run in separate `uvx` environments.
The optional senko diarizer runs inside scribe's environment.
The default models download on first use. The evaluated default model downloads did not require an account.

## Defaults and evaluation limits

The [evaluation harness](../eval/README.md) records a limited historical comparison on eight four-speaker AMI meetings.
It does not establish accuracy for every 2–6-person meeting, microphone or language.

- Nemotron with the historical 120/15 s parakeet baseline reached cpWER 31.64% versus senko's 41.44% on AMI SDM.
  The IHM-mix comparison was 27.39% versus 30.48%. Other metrics and reference styles gave different rankings.
- Current scribe uses parakeet windows of 30 s with 5 s overlap, not the historical 120/15 s baseline.
  Shorter windows reduced dropped speech in that experiment. They did not remove all omissions.
- When word timestamps exist, scribe assigns speakers per word rather than per sentence.
  Brief replies, overlapping speech and inaccurate timestamps can still receive the wrong speaker.

Historical runtimes were hardware- and load-specific. They are not turnaround guarantees.
Review the transcript and speaker labels before relying on them. Speaker labels are not verified identities.

## Requirements

- **macOS with Apple Silicon** (parakeet-mlx requires MLX)
- Python 3.13
- [uv](https://docs.astral.sh/uv/) (used to run parakeet-mlx and mlx-audio)
- [ffmpeg](https://ffmpeg.org/) (all input is normalized to 16kHz mono WAV):
  `brew install ffmpeg`

## Install

From the repository root:

```sh
cd scribe
uv sync
```

## Usage

```sh
uv run scribe examples/meeting.wav
uv run scribe examples/meeting.mp4
```

Use your own permitted inputs at those relative paths.
Scribe writes a neighboring `.txt` file by default, such as `examples/meeting.txt`.
The following transcript is invented:

```text
SPEAKER_00: Thank you for joining. Let us start with the agenda.

SPEAKER_01: I will cover the first item.
```

### Transcript only (no diarization)

```sh
uv run scribe examples/meeting.wav --no-diarize
```

### Options

```sh
uv run scribe examples/meeting.wav -o transcript.txt      # custom output path
uv run scribe examples/meeting.wav -o -                  # write to stdout
uv run scribe examples/meeting.wav --format json         # JSON with timestamps
uv run scribe examples/meeting.wav --format json -o -    # JSON to stdout
uv run scribe examples/meeting.wav --no-diarize          # transcript without speakers
uv run scribe examples/meeting.wav --diarizer senko      # alternative diarizer
```

### JSON output

```sh
uv run scribe examples/meeting.wav --format json
```

Writes `examples/meeting.json` with timestamps and speaker attribution:

```json
{
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "text": "Thank you for joining.",
      "start": 0.531,
      "end": 2.874
    }
  ]
}
```

## Python interface

```python
from pathlib import Path
from scribe import transcribe_audio

result = transcribe_audio(Path("examples/meeting.m4a"), include_embeddings=True)
segments = result.segments
embeddings = result.embeddings
```

`transcribe_audio(original_path, *, diarizer=DEFAULT_DIARIZER, include_embeddings=False, on_step=None, on_log=None)` accepts an original audio path.
It returns `AudioTranscription`, with attributed `MergedSegment` values and a speaker-keyed `SpeakerEmbedding` map.
Each embedding has `centroid`, `speech_s`, and `n_subsegments` fields.
The map is empty when embeddings are disabled.
A centroid can be `None` when no usable voice window exists.
Speaker labels remain cluster labels, not verified identities.

Scribe normalizes the source once to a temporary WAV and launches recognition alongside diarization.
With embeddings enabled, it computes voiceprints after diarization while recognition can still run.
It then collects recognition and merges the word-level timestamps with speaker turns.
The temporary WAV stays available until every submitted worker stops, including on failure.
Scribe removes it before returning or propagating an error and leaves the original untouched.

`on_step(index, total)` reports phase starts, not completion: `0` preparing, `1` recognition launch, `2` diarization, `3` embeddings, `4` merge.
The total is always `5`. Phase `3` is omitted when embeddings are disabled.
A callback sequence is not a success result. Callers mark completion only after the operation returns.

Recognition collection uses the existing 600-second future timeout.
This is **not a hard return deadline**: executor shutdown still waits for the worker after a timeout or another failure.
There is no separate diarization or embedding timeout, retry, or cancellation policy.
Diarization and embedding errors remain primary during unwinding. Recognition errors propagate when Scribe collects recognition.
Audio conversion and model failures retain `ScribeError` diagnostics.

`on_log(message)` enables model counts and warnings for standalone output.
After a diarization error, a supplied logger also receives secondary recognition diagnostics through the existing 30-second collection timeout.
These diagnostics do not replace the diarization error or remove the wait for worker shutdown.
Without `on_log`, preparation and the pipeline emit no log messages.
Diane passes no logger and simply waits while unwinding.
Embedding failures do not trigger secondary recognition diagnostics.

The standalone diarized CLI uses this operation with embeddings disabled and retains its text/JSON output.
`--no-diarize` uses Scribe's transcript-only path, with preparation and cleanup still internal to Scribe.
The low-level `run_transcription`, `diarize`, and `merge` functions remain available for evaluation on already prepared audio.
Applications should use `transcribe_audio` rather than prepare temporary audio or reconstruct this execution order.
Diane owns its timestamp layout, names, metadata, and transcript/voiceprint-sidecar persistence.

## Development

```sh
uv sync --dev
uv run pytest -q
uv run ruff check
```

## License

MIT for the software. Preserve [LICENSE](LICENSE) and [NOTICE](NOTICE), which identify the upstream source and local modifications.
Models, datasets, recordings and generated outputs have separate terms.
The Diane source checkout's [licensing inventory](../docs/licensing/index.md) records evidence, obligations and unresolved gates.
