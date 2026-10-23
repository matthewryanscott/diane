# Diane

*"Diane, make a note of..."*

Diane transcribes recurring meetings locally, separates speech by speaker, and helps you assign speaker names.
It writes timestamped transcripts and voiceprint sidecars to `transcripts/`.
Names you assign or confirm can help identify speakers in later meetings.

The intended use is recurring meetings with 2–6 people.
This is not an accuracy guarantee. Check important words, speaker boundaries, and names against the audio.
Diane does not generate meeting summaries.

## Documentation

Choose a guide for your goal:

- **Learn:** [Your first meeting transcript](docs/tutorial.md). Install Diane in a fresh checkout and transcribe a non-sensitive, synthetic two-voice meeting.
- **Do:** [Meeting workflows](docs/how-to.md). Process recurring recordings, name and review speakers, find transcripts, and solve workflow problems.
- **Look up:** [Reference](docs/reference.md). Commands, discovery rules, file formats, models, requirements, and errors.
- **Understand:** [How Diane works](docs/explanation.md). Local inference, downloads, speaker identity, enrollment, privacy, and model tradeoffs.

See the [evaluation guide](eval/README.md) for measurements and their limits.
See the [licensing inventory](docs/licensing/index.md) for provenance, redistribution obligations, and unresolved gates.

## Before you start

Use macOS on Apple Silicon, Python 3.13, [uv](https://docs.astral.sh/uv/), and the FFmpeg tools.
Installation and initial model downloads can need network access.
Transcription, diarization, and voiceprint matching run locally.

Run Diane from the workspace containing `Recordings/` and `transcripts/`.
In a source checkout, use its root with `uv run diane`; for an installed wheel, enter your data workspace and invoke the environment's `bin/diane` directly.
Paths follow the startup working directory, never the package install directory.

`Recordings/` is the source directory, often a symlink to a live Voice Memos store.
**Keep that directory and its symlink target read-only.**
Diane scans dated filenames, not the Voice Memos database.
The tutorial uses a separate checkout and synthetic copies instead of live recordings.

Recordings, transcripts, and voiceprint sidecars can contain sensitive data.
Local processing does not make those files safe to share or commit.
