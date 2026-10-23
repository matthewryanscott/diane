# Reference

This page describes Diane's commands, file formats, and Python interfaces.
Examples use `uv run diane` from a source checkout's root; that directory is the workspace.
`ID` means a recording or transcript ID, according to the command's selection rules below.

## Runtime and dependencies

| Requirement | Current contract |
| --- | --- |
| Platform | macOS on Apple Silicon. Scribe's Parakeet recognizer requires MLX. |
| Python | `>=3.13,<3.14`, for Diane and the included Scribe package. |
| Environment runner | `uv`, including `uvx`. Scribe launches isolated model environments with `uvx`. |
| Audio tools | `ffmpeg` for normalization, `ffprobe` for recording duration, and `ffplay` for interactive speaker playback. |
| Diane dependencies | `cyclopts>=3`, `httpx>=0.27`, `pyyaml>=6`, `rich>=13`, and the local `scribe` package. |
| Scribe dependency | Senko pinned to commit `40250276a691489786bd20bbebffa609dbf5acb0` from `narcotic-sh/senko`. |

A cold build of pinned Senko also needs a native C++20 toolchain.
Its [CMake declaration](https://github.com/narcotic-sh/senko/blob/40250276a691489786bd20bbebffa609dbf5acb0/CMakeLists.txt) requires CMake 3.18 or newer and Python's interpreter and development module.
It sets a macOS deployment target of 14.0. That declaration does not establish support for the whole stack on every macOS 14 installation.
The [walkthrough](tutorial.md#walkthrough-evidence) reused cached dependencies and did not verify a cold native rebuild.

Diane captures the shell's current working directory at process startup as its workspace (`diane.db.PROJECT_ROOT`).
Discovery reads `Recordings/`, and generated transcripts and voiceprint sidecars live in `transcripts/`, all under that workspace.
`slice` scans workspace-wide `.txt` files, not the package's install directory.
There are no command-line directory overrides, parent-directory searches, or package-location fallbacks.

For an installed wheel, enter your data workspace and invoke the installed executable directly:

```sh
cd /path/to/meeting-workspace
/path/to/environment/bin/diane recording list --limit 25
/path/to/environment/bin/diane transcribe all --dry-run
```

This intentionally corrects the former package-relative behavior: invoking Diane from another directory now selects that directory, not the source checkout or `site-packages`.
Run from the checkout root to retain the source workflow; run from your data workspace for an installed wheel.
Python callers share the workspace captured when `diane.db` is first imported; changing directories after import does not reconfigure these paths.

**Keep `Recordings/` and its symlink target read-only.**
Discovery does not query the Voice Memos database.
Transcription reads original audio and writes generated files outside the source directory.
Recordings, transcripts, and voiceprint sidecars can contain sensitive data. Local processing does not make them safe to publish.

### Models and execution

| Operation | Implementation and settings |
| --- | --- |
| Speech recognition | `uvx 'parakeet-mlx>=0.4,<1'`, with JSON output, 30-second chunks, and 5-second overlap. The wrapper does not set a model ID. Parakeet selects its default. |
| Default diarization | `mlx-community/Nemotron-3-Diarization`, loaded through isolated `mlx-audio==0.5.6` on Python 3.13. Nonstreaming inference (`"offline"` model mode), threshold `0.5`, frame length `0.01` seconds. This mode does not disable downloads. |
| Alternative diarization | Senko, using `Diarizer(device="auto")`. Selectable through Scribe, not Diane's CLI. |
| Voiceprints | Senko CAM++ embeddings, including when Nemotron supplies the speaker turns. Diane enables these on every transcription. |

The [historical evaluation](../eval/README.md) identifies Parakeet 0.5.2's default as `mlx-community/parakeet-tdt-0.6b-v3`.
That records the evaluated model, not a model-ID pin in the current wrapper.

Scribe converts every input to a temporary 16 kHz, mono, signed 16-bit PCM WAV with FFmpeg.
Recognition runs concurrently with diarization. Embedding extraction follows diarization while recognition can still run.
The temporary WAV remains available until the recognition worker stops, including on failure, and is then removed.
Scribe also removes its temporary model-output directories. It leaves the original audio unchanged.

Dependency installation and initial model downloads may require network access.
The evaluated default model downloads did not require an account, but that is not an authentication guarantee for every dependency or future version.
Recognition, diarization, and voiceprint matching execute locally.

Diane and Scribe do not define a model cache directory or expose cache/download flags.
`uvx` manages its tool environments, and the model dependencies manage downloaded weights.
The repository does not fix their cache locations or guarantee that a cache is complete.
An existing installation is not a promise of offline operation: missing environments or model files can still require downloads.
There is no Diane cache purge, model prefetch, offline-mode, or download-retry command.
See the [licensing inventory](licensing/index.md) for provenance, obligations, and unresolved use gates.

## Diane command surface

Use `uv run diane --help` or a command's `--help` for parser help.
The following table lists the application-defined arguments and options.

| Command | Arguments and options | Selection and effect |
| --- | --- | --- |
| `recording list` | `--limit INT`, default `25` | Shows discovered recordings newest first, including duration and a transcript status marker. A limit of zero or less shows all. |
| `transcribe ID` | `--dry-run`, default false | Resolves one recording prefix. Checks its exact canonical transcript path. Existing output is left unchanged. Otherwise creates a transcript and voice sidecar. |
| `transcribe all` | `--dry-run`, default false | Processes pending discovered recordings newest first. Skips exact IDs already represented by transcript filenames and recordings whose audio is absent. |
| `transcript list` | `--json`, default false | Lists direct `.txt` entries in reverse filename order. The table retains orphan transcripts but leaves recording-derived date, duration, and title blank. JSON uses frontmatter instead. |
| `transcript path ID` | One required ID | Prints an absolute transcript path for a unique, case-insensitive filename-ID prefix. Does not require source audio or recording discovery. |
| `transcript absolute ID` | One required recording ID | Requires the recording and its exact canonical transcript. Rewrites relative offsets as local timestamps. Already-absolute transcripts remain unchanged. |
| `transcript absolute all` | None | Converts associated transcripts in ascending filename order. Also attempts the limited precision adjustment described below for already-absolute transcripts. |
| `speakers ID [ID ...]` | One or more recording IDs | Requires each recording and canonical transcript. Plays snippets and prompts for remaining anonymous labels, including labels skipped previously. |
| `speakers all` | None | Prompts for remaining anonymous labels in associated transcripts, in reverse filename order. Skips transcripts without a matching recording. |
| `speakers audit [ID ...]` | Zero or more filename-ID prefixes | Without IDs, considers all transcripts. With IDs, filters by prefix rather than requiring a unique match. Reviews only unconfirmed automatic names, newest filename first. Skips orphan recordings. |
| `slice` | Required `--start ISO` and `--end ISO` | Prints timestamped lines within the inclusive range, sorted by time. Scans workspace-wide `.txt` files with `timestamps: absolute`, excluding hidden directories. |

There is no Diane option for output format, output path, model selection, diarizer selection, speaker count, summaries, or calendar delivery.
Use Scribe's separate command for its supported format and diarizer options.

### Preview, output, and failures

`--dry-run` discovers recordings and may probe audio duration with `ffprobe`.
It does not run transcription models or write transcripts or sidecars.
Single-recording preview still checks for an existing canonical transcript first.

`transcript path` writes its path to stdout and reports lookup errors on stderr with exit status 1.
`transcript list --json` writes JSONL to stdout without a table.
Transcription progress and speaker prompts use stderr, although interactive input also uses terminal stdout.
`slice` writes matching lines to stdout, with a blank line after each result and `[parent-directory]` after each timestamp.

An unresolved or ambiguous recording ID, missing audio, or missing required canonical transcript can prevent a single command from proceeding.
Manual naming warns and skips unresolved IDs when other requested IDs remain usable.
Batch transcription catches per-recording exceptions, continues, and prints a failure summary.
It does not explicitly set a failing exit status for those caught errors. Do not treat its exit status as proof that every recording succeeded.
There is no automatic transcription retry or identity repair.

### Interactive naming

Snippets loop through `ffplay`, with each snippet capped at 15 seconds.
Tab selects the next snippet when several exist. Enter submits the typed answer.

| Prompt | Answer | Result |
| --- | --- | --- |
| Manual naming | Empty | Skips that anonymous speaker. It remains available in a later pass. |
| Manual naming | A valid person name | Replaces the anonymous label and records `source: manual` when that cluster has a sidecar entry. |
| Audit | Empty or unchanged proposed name | Records `source: confirmed` and retains the match score. |
| Audit | A valid different name | Safely corrects the label when attribution is known and unique. Records `source: manual` and clears the score. |
| Either | An anonymous `SPEAKER_` plus digits label | Does not enroll a person or resolve that pending target. |
| Either | A name that cannot round-trip as one transcript label | Rejects that answer without applying it. Other safe answers can still succeed. |

Names containing colons or newlines cannot round-trip as transcript speaker labels.
A shared display name combines turns from several original labels. Playback does not isolate one original cluster.
Audit can confirm a shared proposal but cannot independently correct an ambiguous cluster.
Missing or invalid provenance also prevents independent corrections to automatic names.
Transcript-only manual naming remains possible without a sidecar. It does not fabricate embeddings or create a sidecar.

### Timestamp conversion and slicing

Relative bodies use `[MM:SS]` or `[HH:MM:SS]` from the recording start.
Absolute conversion uses the discovered recording date in local time, not an independently edited frontmatter start time.
It writes body timestamps as `[YYYY-MM-DDTHH:MM:SS]`, without a timezone suffix, and sets `timestamps: absolute`.
The `recorded` metadata retains the local ISO timestamp with its offset.

`transcript absolute all` also handles limited precision differences in already-absolute transcripts.
It shifts body timestamps only when old and newly discovered start times match at minute precision and differ by at least one second.
Different-minute values remain unchanged. If `recorded` is missing, it adds that metadata without shifting the body.
This is not general timestamp repair.

`slice` accepts values parsed by Python's `datetime.fromisoformat`, such as `2026-04-05T14:00`.
Its comparisons use local wall-clock values. Supplied timezone offsets are removed, not converted to another timezone.
It includes a line when that line's timestamp falls within `[start, end]`, not when a whole speaker turn overlaps the range.
The scan is not limited to `transcripts/`, and the parent-directory label is not a unique file identifier.

## Recording discovery and transcript association

Only direct, nonhidden files in `Recordings/` participate. Discovery does not filter by extension.
FFmpeg must be able to decode an input for transcription to succeed.

| Filename stem | Recognition |
| --- | --- |
| `YYYYMMDD HHMM Title` | Date-first form, checked first. |
| `Title YYYYMMDD HHMM` | Title-first form, checked only when date-first does not match. |
| Undated, date-only, or title-first with trailing counters | Not recognized. |

For example, both `20260405 1400 Voice Chat.m4a` and `Voice Chat 20260405 1400.m4a` match.
These are synthetic names, not repository recordings.
A later date-like phrase remains in a date-first title.
Do not rename source recordings to change discovery.

Diane prefers file creation time (`st_birthtime`) for the recording date.
If unavailable, it parses the filename's date and minute in local time.
Duration comes from `ffprobe`. Probe failures produce a zero duration rather than a discovery error.

The full recording ID is the SHA-256 hex digest of the filename stem, including its case and spacing.
The displayed `short_id` is its first eight characters.
Recording lookup lowercases a supplied prefix and matches against these eight-character IDs.
It rejects missing and ambiguous matches.

The canonical transcript path is `transcripts/{local YYYYMMDDTHHMMSS}-{short_id}.txt`.
The corresponding voice sidecar replaces `.txt` with `.voice.json`.
Single-recording transcription checks only that canonical path, not alternate filenames with the same ID.

Transcript association uses the filename, not YAML `recording_id`:

| Policy | Filename parsing | Callers |
| --- | --- | --- |
| `suffix-or-stem` | ID is everything after the first hyphen. Without a hyphen, use the whole stem. | Transcript path lookup, batch skip/status, speaker audit, and JSON fallback. |
| `delimited-only` | Same suffix rule, but an undelimited stem has an empty ID. | Transcript table association, bulk manual naming, bulk timestamp conversion. |

Association and batch skip checks compare exact, case-sensitive IDs.
Transcript path lookup alone uses case-insensitive prefix matching and requires a unique transcript.
A transcript may remain visible or resolvable after its source recording disappears.
A renamed same-ID transcript can skip batch transcription while a single-recording command still requires the canonical path.
Commands do not rename transcripts to reconcile these differences.

## Transcript and JSONL schemas

A Diane transcript is UTF-8 text with YAML frontmatter delimited by `---`, followed by timestamped speaker turns.
New transcripts contain these fields:

| YAML field | Value |
| --- | --- |
| `recording_id` | Eight-character recording ID. |
| `recorded` | Recording start as a local ISO datetime string, including timezone offset. |
| `duration` | Audio duration in seconds. A failed duration probe can produce `0.0`. |
| `title` | Title parsed from the source filename. |
| `speakers` | Unique final body labels in first-occurrence order, including anonymous or unknown labels. |
| `timestamps` | Added as `absolute` by timestamp conversion. New relative transcripts omit it. |

A body line has the form `[MM:SS] SPEAKER_00: speech text`, or its hour/absolute equivalent.
Offsets truncate fractional seconds. Speaker changes always start a paragraph.
Within one speaker's turn, a sentence boundary permits a break after a two-second pause or a two-minute block.
`UNKNOWN` means Scribe could not attribute that speech. It is not a verified identity or a manual anonymous naming target.

`transcript list --json` emits one JSON object per line, not a JSON array:

| JSON key | Type and fallback |
| --- | --- |
| `id` | Frontmatter `recording_id` when truthy, otherwise the `suffix-or-stem` filename ID. |
| `recorded` | Frontmatter value converted to a string, or `null` when absent. |
| `duration_seconds` | Frontmatter `duration` converted to a float, or `null` when absent. |
| `title` | Frontmatter title, or `""`. |
| `speakers` | List from frontmatter speakers, or `[]`. |
| `path` | Transcript path string. The CLI's absolute workspace root makes this absolute. |

JSONL does not discover recordings to fill missing metadata.
The presentation `id` can differ from the filename ID used for association and batch skips.
The reader does not validate or repair arbitrary YAML field types. Invalid YAML or failed numeric conversions remain explicit errors.

## Voice sidecar schema and provenance

Each `{stem}.voice.json` contains one JSON object:

| Top-level field | Current written value |
| --- | --- |
| `version` | `1`. The reader does not enforce this field as a migration gate. |
| `embedding_model` | `"senko CAM++"`. |
| `diarizer` | `"nemotron"`. |
| `origin` | `"transcription"`. |
| `clusters` | Array of cluster objects. Can be empty. |
| `provenance` | Versioned attribution object, or `null` for sidecars without that evidence. |

Each cluster has these fields:

| Cluster field | Meaning |
| --- | --- |
| `speaker` | Immutable original diarizer label. |
| `centroid` | Array of embedding floats. Diane omits clusters whose returned centroid is `None`. |
| `speech_s` | Diarized speech seconds, rounded to two decimals at Diane's transcription boundary. |
| `name` | Current assigned display name, or `null`. |
| `source` | `auto`, `manual`, `confirmed`, or `null`. |
| `score` | Automatic cosine match score, rounded to four decimals, or `null`. Manual correction clears it. Confirmation retains it. |

Scribe's `n_subsegments` count is available through Python, but Diane does not persist it in the sidecar.
A body speaker can have no embedding and therefore no cluster entry.
Missing sidecars load as transcript-only documents. Malformed JSON, malformed cluster arguments, and filesystem errors are not silently ignored.
The reader defaults missing top-level descriptive fields to the values shown above, and missing `clusters` to an empty array.

New sidecars store `provenance` with `version: 1`, a `labels` object, and a `fingerprint` string.
`labels` maps every original label to its current display label, including body-only speakers, `UNKNOWN`, and all actual clusters.
The SHA-256 fingerprint covers that complete map and the ordered speaker-label occurrences in the body.
The input is UTF-8 JSON with sorted keys, compact separators, and non-ASCII characters preserved.
Speech text and timestamps are not fingerprint inputs.
Timestamp conversion and speech-text edits therefore do not invalidate otherwise consistent evidence.

Missing, partial, inconsistent, unsupported-version, or stale provenance means attribution is unknown.
Cluster entries alone do not prove which displayed turns belong to an original cluster.
Decisions use the original groups for the whole pass and apply safe unique-label replacements simultaneously.
A correction that merges display names makes later independent corrections ambiguous.
The identity owner never reconstructs original attribution from an edited legacy body.

Voiceprint enrollment uses named `manual` and `confirmed` clusters only. Automatic names never enroll.
Each cluster contributes equal weight after L2 normalization. The aggregate is normalized again for cosine matching.
Matches require a score of at least `0.86`. Several clusters may match the same person.
Enrollment scans sidecars even when the corresponding transcript was removed, excluding the target transcript during new matching.

### Persistence errors

A successful identity save derives `speakers` metadata from final body labels and preserves unrelated metadata, timestamps, speech text, centroids, and durations.
An unchanged existing document does not write or normalize either file, including blank-only or fully rejected manual passes.

Saving writes the transcript first, then its sidecar. The pair is **not a transaction**.
If the sidecar write fails after the transcript succeeds, `PartialIdentityWriteError` exposes `transcript_path` and `sidecar_path` and chains the original error.
The transcript may contain accepted changes while its sidecar remains stale or unavailable.
There is no hidden retry, rollback, or automatic repair. This exception is not a successful save.
For ambiguous attribution, preserve the existing files privately and produce a fresh transcription with provenance before independent corrections.
No command automatically performs that replacement or recovery.

## Scribe command surface

Scribe is the included audio pipeline package. Its CLI accepts an audio/video path directly, without Diane's filename discovery.
Use `uv run scribe AUDIO [options]` from the root environment, or from the Scribe project after installing its environment.

| Argument or option | Default and behavior |
| --- | --- |
| `AUDIO` | Required input path. Any format FFmpeg can decode. |
| `-o PATH`, `--output PATH` | Default is the input path with `.txt` or `.json` suffix. `-` writes to stdout. An existing output path is overwritten. |
| `--format text\|json` | Default `text`. Controls serialization and the default output suffix. |
| `--no-diarize` | Default false. Runs recognition without speaker attribution. |
| `--diarizer nemotron\|senko` | Default `nemotron`. Applies only when diarization runs. |
| `-h`, `--help` | Prints argparse help. |

Diarized text joins consecutive same-speaker segments into `SPEAKER: text` paragraphs, without Diane's timestamp layout or YAML.
Diarized JSON has `speakers` and `segments` arrays.
Each segment has `speaker`, `text`, `start`, and `end`. Times are relative seconds rounded to three decimals.
The speakers array is sorted, not first-occurrence ordered.
Without diarization, text has no labels. JSON has only `segments`, whose entries omit `speaker`.
Scribe's CLI does not request embeddings and does not create Diane voice sidecars or assign person names.

Missing input paths terminate with an error.
FFmpeg, recognizer, and diarizer failures retain `ScribeError` diagnostics, which the CLI reports as errors.
Other exceptions are not all converted into `ScribeError`.
See [Scribe's README](../scribe/README.md) for standalone usage and pipeline details.

## Public Python interfaces

### Transcription

`diane.transcribe.transcribe_recording(recording, *, on_step=...)` returns `TranscriptionResult(segments, clusters)`.
It passes the original recording path to Scribe with `include_embeddings=True`.
Diane retains original labels, drops null centroids, and rounds speech seconds to two decimals.
`format_diarized_transcript(segments)` produces Diane's relative-timestamp body layout.

`scribe.transcribe_audio(original_path, *, diarizer=DEFAULT_DIARIZER, include_embeddings=False, on_step=None, on_log=None)` returns `AudioTranscription(segments, embeddings)`.
Each `MergedSegment` has `speaker`, `text`, `start`, and `end`.
The embeddings map is keyed by original speaker label.
Each `SpeakerEmbedding` has `centroid`, `speech_s`, and `n_subsegments`. A centroid can be `None`.
Disabled embeddings produce an empty map.

The callback reports phase starts as `(index, 5)`: preparation `0`, recognition launch `1`, diarization `2`, embeddings `3`, merge `4`.
Phase `3` is absent when embeddings are disabled. Only a returned result indicates success.
Recognition collection uses a 600-second future timeout, not a hard return deadline.
Executor shutdown still waits for recognition after timeout or another failure.
There is no separate diarization/embedding timeout, retry, or cancellation policy.
Diarization and embedding failures remain primary during unwinding.
With `on_log`, a diarization failure also collects secondary recognition diagnostics with a 30-second timeout.
Embedding failures do not trigger those diagnostics. Diane supplies no logger.
The temporary WAV remains valid until submitted workers stop and is removed before return or propagated failure.

Scribe retains `run_transcription`, `diarize`, and `merge` for evaluation on prepared audio.
Applications should use `transcribe_audio` rather than reconstruct preparation or execution order.
See the [Scribe Python interface](../scribe/README.md#python-interface).

### Identity

`diane.voiceprints` owns identity state and transcript/sidecar consistency.
The CLI and `diane.speakers` retain prompts, playback, snippets, and timestamp alignment.

| Interface | Contract |
| --- | --- |
| `new_identity(path, frontmatter, raw_body, clusters, *, transcripts_dir=None)` | Returns an unsaved `IdentityDocument`, capturing original labels before automatic naming. Enrollment defaults to `path.parent` and excludes the target transcript. |
| `load_identity(path, recording=None)` | Reads without writing. An optional `Recording` supplies metadata only when frontmatter is absent. |
| Document attributes | `path`, `frontmatter`, `body`, optional `sidecar`, and initial `matches` keyed by original label, with `(name, score)` values. |
| `manual_targets` | Remaining anonymous body labels, including labels without embeddings. |
| `audit_targets` | Only unconfirmed automatic clusters. |
| `IdentityTarget` | Frozen value with immutable answer `key`, display `label`, optional `score`, `speech_s`, and `shared`. |
| `apply_manual(answers)`, `apply_audit(answers)` | Stage explicit `dict[str, str]` answers keyed by original target keys. Unanswered targets remain unchanged. Return frozen `IdentityRejection(key, label, reason)` values. |
| `save()` | Returns `True` after writing a new document or accepted changes, otherwise `False`. Can raise `PartialIdentityWriteError` after a successful transcript write. |

### Association

`diane.db.Recording` contains `unique_id`, `date`, `duration`, `title`, and relative audio `path`, with the `short_id` property.
`Transcript` contains `path`, `filename_id`, and an optional matching `Recording`.

| Interface | Contract |
| --- | --- |
| `get_recordings()` | Discovers recordings newest first without changing source files. |
| `find_recording(prefix)` | Resolves a unique short-ID prefix or raises `ValueError`. |
| `canonical_transcript_path(recording, directory)` | Builds a canonical path. Does not find renamed transcripts. |
| `list_transcripts(directory, *, id_policy="suffix-or-stem", include_recordings=False)` | Returns unsorted direct `.txt` entries. Does not discover recordings by default. Enrichment joins exact filename IDs and retains orphans. Missing/empty transcript directories never trigger discovery. |
| `find_transcript(prefix, directory)` | Resolves a unique case-insensitive filename-ID prefix without discovery. Raises `FileNotFoundError` if absent, `ValueError` if ambiguous. |
| `find_recording_transcript(prefix, directory)` | Resolves the recording first, then requires its exact canonical transcript. |
| `transcript_id(path, *, policy="suffix-or-stem")` | Implements the filename policies described above without reading frontmatter. |
| `transcribed_ids(directory)` | Returns the exact `suffix-or-stem` ID set used for batch status and skipping. |
| `pending_recordings(directory)` | Returns discovered recordings without an exact transcript ID and with source audio present. |

## Source of truth

This reference follows the current implementations in [CLI](../diane/cli.py), [discovery and association](../diane/db.py),
[frontmatter](../diane/frontmatter.py), [timestamps](../diane/timestamps.py),
[transcription](../diane/transcribe.py), [speaker interaction](../diane/speakers.py), [identity](../diane/voiceprints.py),
and [Scribe](../scribe/scribe.py), with dependencies declared in the two `pyproject.toml` files.
