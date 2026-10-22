# How-to guides

Use these procedures for recurring meetings after setup. Run commands from the workspace containing `Recordings/` and `transcripts/`.
Examples use `uv run diane` in a source checkout's root. With an installed wheel, enter your data workspace and replace that prefix with the installed executable, such as `/path/to/environment/bin/diane`.
Replace `ID` with a recording or transcript ID shown by the relevant list command.
Use a longer prefix if a command reports an ambiguous ID.

Keep `Recordings/` and its symlink target read-only. Keep transcripts and voiceprint sidecars private, too.

## Process new recordings after each meeting

Preview pending work before each batch:

```sh
uv run diane recording list --limit 25
uv run diane transcribe all --dry-run
```

Check the titles and dates against the meetings you expect. Then process the pending recordings:

```sh
uv run diane transcribe all
```

`all` processes discovered recordings without matching transcript filename IDs. It does not process every source file.
To select one meeting instead, use its recording ID:

```sh
uv run diane transcribe ID --dry-run
uv run diane transcribe ID
```

A dry run does not run transcription or write transcripts. Review important words and speaker boundaries against the audio after transcription.

If a batch reports failures, address each reported cause before running it again.
Successful transcripts remain available. The next batch skips their filename IDs.
A progress phase announces work starting, not successful completion.

## Make an exported recording discoverable

Diane scans direct, nonhidden files in `Recordings/`. It does not search subdirectories or query the Voice Memos database.
Use either filename form for **copies you control**:

- Date first: `20260405 1400 Voice Chat.m4a`
- Title first: `Voice Chat 20260405 1400.m4a`

An undated name, a date without a time, or `Voice Chat 20260405 1400 2.m4a` will not match.
Date-first matching takes priority when a name fits both forms.

If a source filename does not match:

1. Export or copy the permitted recording into a separate workspace's ordinary `Recordings/` directory.
2. Give that copy a supported filename.
3. Run `uv run diane recording list --limit 0` in that checkout, or the installed `diane recording list --limit 0` executable from that workspace.
4. Check the listed title and date before transcription.

Do not rename originals, write into a live recording-store symlink, or replace that symlink for this procedure.
There is no CLI option to select a different recordings directory.

Diane prefers file creation time over the filename date. A new copy can therefore show the copy date.
Preserve creation time when your export or copy method supports it. Do not assume the filename overrides it.
Recording IDs derive from filename stems, so keep imported copy names stable between runs.

## Assign names and enroll known speakers

Use manual naming for anonymous labels such as `SPEAKER_00`:

```sh
uv run diane speakers ID
```

Listen to the looping snippet before typing a name. Press Tab for another snippet when one is available.
Press Enter without a name to skip that speaker. Run the command again to name skipped speakers.
Existing names do not prevent Diane from offering the remaining anonymous speakers.

To name remaining anonymous speakers across associated transcripts:

```sh
uv run diane speakers all
```

Use consistent names across meetings. Do not type a name unless the audio supports that identity.
Names must fit one transcript speaker label. Names containing a colon or newline are rejected.
Answers such as `SPEAKER_01` remain anonymous and do not enroll a person.

Manual naming updates the transcript and any existing matching voiceprint cluster.
A manually named cluster with an embedding contributes to later voiceprint matches.
A missing sidecar still permits transcript-only naming, but typing a name creates neither a sidecar nor an embedding.

## Review automatic name proposals

Later transcriptions can apply names from previously enrolled voiceprints. These are proposals, not confirmations.
Review one meeting with:

```sh
uv run diane speakers audit ID
```

To review pending automatic names across transcripts, omit the ID:

```sh
uv run diane speakers audit
```

At each audit prompt:

- Press Enter, or type the unchanged proposed name, to confirm it.
- Type a different name only when you can identify the speaker and Diane can safely apply the correction.
- If unsure, stop with Ctrl-C. Unlike manual naming, Enter in an audit does **not** skip the proposal.

Confirmation keeps the match score and makes the cluster eligible for enrollment.
An accepted correction records a manual name and clears the old match score.
Automatic proposals never contribute to enrollment until you confirm or correct them.
Manual and confirmed names do not appear in later audits.

Check rejection messages before assuming a correction succeeded. Other safe answers in the same pass can still save.
A blank-only manual pass or a fully rejected pass does not write either file.

## Handle duplicate names, legacy transcripts, and stale identity evidence

Several original speaker clusters can share one displayed name.
When Diane warns about shared playback, the snippets combine those clusters' turns. They do not isolate the current cluster.
Do not use combined playback to guess an independent correction.

Use these rules:

- Confirm a shared proposal only if you have evidence that the named identity is correct for the cluster under review.
- Do not attempt to split a shared displayed name through repeated audit corrections. Diane rejects ambiguous independent corrections.
- With valid provenance, a correction from a unique label can merge it into another displayed name.
  That merge makes later independent corrections ambiguous.
- With missing, partial, inconsistent, or stale provenance, audit can confirm the unchanged name but rejects a different name.
- In that legacy or stale state, manual naming can still replace an unambiguous raw anonymous label.
  Diane rejects a name that collides with another existing label or another answer in the same pass.

A sidecar alone does not prove which transcript turns belong to a named cluster.
Changing labels or their occurrence order can invalidate provenance. Timestamp-only conversion and speech-text edits do not invalidate it.
Do not edit raw transcript labels, sidecar names, or provenance fields to bypass rejection.

### Produce fresh evidence when a correction is unsafe

There is no force-transcription or automatic identity-repair command.
To replace an ambiguous result without guessing:

1. Locate the existing transcript with `uv run diane transcript list` and `uv run diane transcript path ID`.
2. Preserve the transcript and its companion `.voice.json` file in a private archive outside the workspace.
3. Move all same-recording transcript copies and their sidecars out of the active `transcripts/` directory.
4. Confirm that the source recording remains discoverable with `uv run diane recording list --limit 0`.
5. Preview `uv run diane transcribe ID --dry-run`.
6. Run `uv run diane transcribe ID` to produce a fresh transcript and provenance.
7. Review the new speaker labels and proposals against the audio.

Use the File column or JSON paths to locate duplicate transcripts when `transcript path` reports ambiguity.
Archive sidecars with their transcripts: enrolled sidecars still contribute even when their transcript is absent.
Keep the archive outside the workspace because `slice` also scans workspace-wide text files.

Fresh provenance does not guarantee that diarization separates every speaker correctly.
If the new result still shares a displayed name, do not invent a per-cluster correction.
The CLI also has no direct correction prompt for names already marked manual or confirmed.

## Locate and read a meeting transcript

List transcripts and resolve the selected ID:

```sh
uv run diane transcript list
uv run diane transcript path ID
```

Open the printed absolute path in your preferred reader. The file contains YAML metadata followed by timestamped speaker turns.
Relative timestamps such as `[03:12]` are offsets from the recording start.
For paths and metadata suitable for scripts, request one JSON object per transcript:

```sh
uv run diane transcript list --json
```

Listings use reverse filename order, not a metadata-date sort.
They retain transcripts whose recordings are unavailable. `transcript path` also works without the source recording.
A table can therefore show a file with blank recording details.

In contrast, `speakers ID` and `transcript absolute ID` require a discoverable recording and its exact canonical transcript path.
A readable transcript is not proof that those commands can operate on it.

## Read what was said during a time range

Convert the selected transcript to local absolute timestamps, then select the interval:

```sh
uv run diane transcript absolute ID
uv run diane slice --start 2026-04-05T14:00 --end 2026-04-05T15:30
```

Replace the example times with your local interval. Seconds are optional.
`absolute` rewrites the transcript. It does not change source audio or speaker identity.

`slice` prints matching lines in time order and tags each line with its parent directory.
It scans workspace-wide `.txt` files with absolute-timestamp metadata, excluding hidden directories.
It does not limit results to the selected meeting or to `transcripts/`.

For multiple meetings, use `uv run diane transcript absolute all` before slicing.
This bulk command also adjusts eligible older absolute timestamps when only their sub-minute precision differs from the recording time.
It does not repair arbitrary recording-date errors.

## Troubleshoot setup and workflow problems

### A required command or model cannot start

Use macOS with Apple Silicon, Python 3.13, and `uv`, including `uvx`.
Make `ffmpeg`, `ffprobe`, and `ffplay` available in the command environment.
On a system with Homebrew, install the FFmpeg tools with `brew install ffmpeg`.

Initial dependency and model downloads can require network access.
A successful dry run does not prove that models, audio decoding, or playback work.
Use the reported error to distinguish installation or download failures from recording discovery problems.
See [Scribe requirements](../scribe/README.md#requirements) for the underlying transcription runtime.

### A meeting is missing from the recording list

Check the direct-file location, hidden status, and filename forms in [Make an exported recording discoverable](#make-an-exported-recording-discoverable).
A missing `Recordings/` directory produces an empty list. Do not repair discovery by modifying the live recording store.

A zero duration is not proof of an empty recording. Duration probing returns zero when `ffprobe` fails.
Check tool availability and the permitted copy's readability before relying on duration totals.

### Batch and single-recording commands disagree about an existing transcript

Batch transcription skips an exact recording ID found in any transcript filename, including a noncanonical filename.
Single-recording transcription checks only its canonical `<local YYYYMMDDTHHMMSS>-<short_id>.txt` path.
Neither command renames transcripts or overwrites an existing canonical transcript.
Use the transcript list and printed paths to identify the files before choosing fresh transcription.
Changing frontmatter `recording_id` does not repair filename-based association.

### Naming or conversion cannot find a recording

Use `transcript path ID` to read the retained transcript without source discovery.
For playback or conversion, restore access to the unchanged source or its established imported copy.
Do not rename the source to fit a transcript, or assume a noncanonical transcript satisfies canonical-path lookup.

### Playback or prompts fail

Run speaker commands in an interactive terminal. They use terminal input and `ffplay` playback.
Check that `ffplay` is available and the associated audio exists before assigning names.
Do not confirm an identity from text alone when playback is unavailable.

### A sidecar cannot be read or an identity save is incomplete

Malformed sidecar JSON and filesystem errors remain errors. Diane does not silently replace them with empty identity data.
Preserve the files privately and address the reported access or storage problem before proceeding.

`PartialIdentityWriteError` means the transcript was written but its sidecar write failed.
Treat the pair as inconsistent, not as a successful save. There is no automatic retry, rollback, or repair.
Do not repair it by manually changing raw identity fields.
Use [fresh transcription](#produce-fresh-evidence-when-a-correction-is-unsafe) when the pair no longer provides trustworthy attribution.
