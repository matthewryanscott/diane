# Your first meeting transcript

In this tutorial, you will install Diane and transcribe a short synthetic meeting with two voices.
You will find its transcript, assign names, and review names in a second recording.

Diane targets recurring meetings with 2–6 people.
This exercise uses two synthetic voices, not a six-person accuracy test.
Use public, synthetic, or explicitly consented non-sensitive material while you learn.

## 1. Start with a separate checkout

Use macOS on Apple Silicon. Install Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), and [FFmpeg](https://ffmpeg.org/download.html).
Homebrew users can install the latter tools with:

```sh
brew install uv ffmpeg
```

A cold Senko build also needs a C++20 compiler.
Check its [declared build tools](reference.md#runtime-and-dependencies) before installation.

Clone this repository into a new directory, then enter that directory.
Do not use a checkout connected to a live Voice Memos store for this exercise.
Run the remaining commands from the new checkout's root.

Check the tools and install the project:

```sh
uv --version
uvx --version
ffmpeg -version
ffprobe -version
ffplay -version
uv sync --python 3.13
uv run python --version
uv run diane --help
```

`uv sync` creates a project environment and installs the bundled Scribe dependency.
Python must be 3.13, not 3.14.
The model tools also use isolated `uvx` environments.
Dependencies and models can download on first use.
A fresh project environment does not clear existing uv or model caches.
No account or API token is required for the default transcription models.

## 2. Make non-sensitive sample audio

Create the audio outside the repository.
The following commands use macOS speech synthesis and FFmpeg.
List the installed voices first:

```sh
say -v '?'
```

Choose two distinct installed English voices.
The walkthrough used `Samantha` and `Daniel`.
If either is unavailable, change the corresponding variable to an installed voice before continuing.
Do not assume a voice download is available offline.

```sh
SAMPLE_DIR="$(mktemp -d /tmp/diane-meeting-sample.XXXXXX)"
VOICE_A='Samantha'
VOICE_B='Daniel'
say -v "$VOICE_A" -r 150 -o "$SAMPLE_DIR/first.aiff" \
  'This is a synthetic meeting for a local transcription check. Our garden project starts on Monday morning. We need to review the schedule and compare the supply list with our tools. I will bring the seeds and a clean container. We will practice outside if the weather stays mild. Please check the speaker labels against this audio.'
say -v "$VOICE_B" -r 150 -o "$SAMPLE_DIR/second.aiff" \
  'I suggest that we check the supply list first. We need a notebook, a pencil, and a container for the seeds. I will bring the notebook and record our decisions. Let us meet again on Tuesday at ten in the morning. We can compare our progress and choose the next task. This is only a synthetic practice recording.'
ffmpeg -nostdin -i "$SAMPLE_DIR/first.aiff" -i "$SAMPLE_DIR/second.aiff" \
  -filter_complex '[0:a][1:a][0:a][1:a]concat=n=4:v=0:a=1[out]' \
  -map '[out]' -ar 16000 -ac 1 "$SAMPLE_DIR/meeting.wav"
```

These four turns alternate voice A, voice B, voice A, and voice B.
Keep the sample directory for comparison with the transcript.

Create a source directory only in this fresh checkout:

```sh
mkdir Recordings &&
  cp "$SAMPLE_DIR/meeting.wav" 'Recordings/20261022 1000 Synthetic planning.wav'
```

**Stop if `Recordings` already exists.** Do not replace a symlink or copy files into a live recording store.
The `mkdir` command must succeed before you run `cp`.
All subsequent Diane commands treat source audio as read-only.

## 3. Discover and transcribe the sample

```sh
uv run diane recording list --limit 25
uv run diane transcribe all --dry-run
```

You should see `Synthetic planning` and an eight-character ID.
The date can reflect the file creation time instead of the date in its filename.
The dry run should show one pending recording.

```sh
uv run diane transcribe all
uv run diane transcript list
uv run diane transcript list --json
```

The first transcription may download models and take longer than later runs.
A successful run writes a `.txt` transcript and a `.voice.json` sidecar in `transcripts/`.
The transcript list should contain one meeting.
The JSON command prints one object per line, not a JSON array.

## 4. Read and name the speakers

Replace `ID` in each following command with the ID from your list.

```sh
uv run diane transcript path ID
```

Open the printed path in a text editor.
Expect YAML metadata followed by turns such as this illustrative excerpt:

```text
[00:00] SPEAKER_00: This is a synthetic meeting for a local transcription check.

[00:19] SPEAKER_01: I suggest that we check the supply list first.
```

Exact words, offsets, and speaker labels can differ.
Compare the transcript with the four known turns.
Check that both voices appear, without treating the labels as person identities.

```sh
uv run diane speakers ID
```

The command plays snippets and prompts for each unnamed speaker.
Use `Practice A` for the voice that discusses the schedule.
Use `Practice B` for the voice that discusses the notebook.
Press Enter instead if a snippet does not let you identify the voice.
Only explicit names with real embeddings can enroll a voiceprint.
A skipped speaker remains available in later naming passes.

Run the list again and inspect the transcript:

```sh
uv run diane transcript list
uv run diane transcript path ID
```

Names should replace the labels you identified.
The sidecar retains the original cluster labels and records your manual decisions.
Do not commit the audio, transcript, or sidecar.

## 5. Review a recurring recording

Copy the same synthetic sample under a new, title-first filename:

```sh
cp "$SAMPLE_DIR/meeting.wav" 'Recordings/Synthetic followup 20261022 1100.wav'
uv run diane transcribe all --dry-run
uv run diane transcribe all
uv run diane transcript list
```

Only the new filename should be pending.
This repeated audio makes a controlled enrollment check, not a test of changed microphones or real recurring meetings.
If the first naming pass enrolled both voices, the new transcript may propose their names automatically.

Replace `FOLLOWUP_ID` with the second recording's ID:

```sh
uv run diane speakers audit FOLLOWUP_ID
```

Listen to the snippets. Press Enter to confirm each correct proposal.
Type the correct name only when the audio supports a correction.
If no automatic proposal exists, the command reports no speakers to review.
Use `uv run diane speakers FOLLOWUP_ID` to name remaining anonymous labels.
See [safe corrections](how-to.md#handle-duplicate-names-legacy-transcripts-and-stale-identity-evidence) before changing shared or legacy names.

You now have two local transcripts and explicit speaker decisions.
Continue with the [meeting workflows](how-to.md), or look up the [file formats and commands](reference.md).

## Walkthrough evidence

This walkthrough completed on **2026-10-22**, using a fresh tracked-source clone and a new `.venv`.
Only tracked source files entered the clone. Ignored recordings, transcripts, and environment artifacts were not copied.

| Environment | Observed value |
| --- | --- |
| Machine | Apple M1 Pro, 32 GiB RAM, arm64 |
| System | macOS 26.7.1, build 25G241 |
| Python | CPython 3.13.5 |
| Tools | uv and uvx 0.13.5, FFmpeg/ffprobe/ffplay 9.0.2 |
| Install | `uv sync --python 3.13` created `.venv` and installed 47 packages |
| Sample | Installed Samantha and Daniel voices at 150 words per minute, four alternating turns, 80.08475 seconds |

Git, uv, FFmpeg, and the selected system voices were already installed.
The Homebrew bootstrap command was not rerun.

Network access was available and was not disabled.
Existing uv package/tool caches were retained.
The Hugging Face cache already contained `mlx-community/Nemotron-3-Diarization` and `mlx-community/parakeet-tdt-0.6b-v3`.
The installed pinned Senko package supplied its model resources.
No network-traffic audit or offline test was performed.
This verifies a clean project environment, not an empty-cache installation.

| Exercised operation | Observed result |
| --- | --- |
| Installation, Diane command help, Scribe help | Completed successfully in the new environment |
| Date-first discovery and batch preview | One pending synthetic meeting |
| First transcription | `.txt` and `.voice.json` written, with two speaker clusters and four alternating turns |
| Single-recording preview and transcription | A third synthetic copy was previewed, transcribed, and resolved by ID |
| Transcript table, JSONL, and path lookup | Located the generated meeting and its metadata |
| Manual naming | Two real terminal prompts completed, with two `manual` sidecar decisions |
| Title-first followup and audit | Only the new copy was pending. Both automatic proposals were explicitly confirmed |
| Existing-output and no-pending commands | Left existing transcripts unchanged and reported no pending work |
| Absolute conversion and time slicing | Converted both transcripts and returned the synthetic turns in the selected day |
| Scribe text, JSON, no-diarization, and senko modes | Wrote or printed the documented formats, including default output naming and plain JSON |
| Missing IDs, missing audio, and invalid audio | Reported lookup/input/decode errors with exit status 1 |

All three source copies retained the sample's SHA-256 after processing.
The repeated sample tests a controlled enrollment path, not changed microphones, overlapping speech, or new participants.
These results do not establish accuracy for all meetings with 2–6 people.

An initial automated naming attempt stalled and left audible looping playback. That attempt was interrupted, not accepted as verification.
The completed naming and audit checks used a pseudo-terminal with raw-mode, per-character echo handshakes.
Their parent environment set `SDL_AUDIODRIVER=dummy`, so playback stayed silent.
Cleanup left no `ffplay`, naming command, or walkthrough-driver processes.
The completed checks verify CLI state transitions, not audible playback quality or human identity review.
