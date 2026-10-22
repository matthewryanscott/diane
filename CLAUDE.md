# Diane

Diane transcribes recurring meetings locally, separates speech by speaker, and supports interactive naming and voiceprint matching.
The intended use is recurring meetings with 2–6 people. Check accuracy against the audio.

## Project guidance

- `Recordings/` supplies source files. Discovery scans date-first `YYYYMMDD HHMM Title.ext` and title-first `Title YYYYMMDD HHMM.ext` filenames, not the Voice Memos database.
- `transcripts/` holds generated text transcripts and `.voice.json` sidecars. Naming, audits, and timestamp conversion update these outputs.
- Read `README.md` for the CLI workflow. Use `transcribe all --dry-run` before a batch transcription.
- Read `eval/README.md` when evaluating transcription or diarization accuracy. Its measurements do not guarantee results on other meetings.
- Processing runs locally. Dependency installation and initial model downloads may need network access.
- Treat recordings, transcripts, and voiceprint sidecars as sensitive data. Keep private content out of examples and commits.

## Critical rules

- **READ ONLY** in symlinked locations (`Recordings/`). Never write, modify, rename, or delete files there. These are live system data stores.
