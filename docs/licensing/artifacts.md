# Artifact checks

This protocol defines release checks. It does not report a current build or smoke result.
Record the source commit, tool versions, artifact filenames, SHA-256 digests and observed outcomes separately.
An sdist cannot contain its own final digest.

## Build and inspect

Build Diane and companion Scribe wheels/sdists from a clean export of the intended commit.
Use `uv build --no-build-isolation --python` with the audited B1 interpreter.
[Dependencies](dependencies.md) records B1's exact tools. The unversioned project backend requirement does not pin future tools.

| Artifact | Required contents and boundary |
| --- | --- |
| Diane wheel | `License-Expression: MIT`, `License-File: LICENSE`, `License-File: THIRD_PARTY_NOTICES` and both files under `.dist-info/licenses/`. Diane modules only. |
| Scribe wheel | `License-Expression: MIT`, `License-File: LICENSE`, `License-File: NOTICE` and both files under `.dist-info/licenses/`. Modified Scribe module only. |
| Diane sdist | Root legal files, Scribe LICENSE/NOTICE and publication source/provenance. No private inputs/reports, raw runs, model caches, recordings, transcripts or installed environment. |
| Scribe sdist | Original MIT LICENSE, local NOTICE and companion source. No models, dependency environments or recordings. |

Inspect archives directly and compare packaged legal bytes with source files.
Inspect `Requires-Dist`: Diane's bare Scribe requirement omits uv's source mapping. Use the audited companion wheel.
Scribe must retain the exact Senko Git direct reference, not a registry namesake.

## Installed boundary

Install both built wheels over the checkout packages in an isolated environment with the frozen root runtime dependencies.
Inspect installed metadata and legal files. Run both CLI help entry points outside the source tree.
Use only permitted synthetic inputs for safe recording/transcript listing checks. Do not access live Voice Memos data or personal transcripts.

A documentation-only rebuild does not require another model run when executable code and locked inputs match the previously exercised candidate.
A carried-forward model result must identify that earlier candidate and establish those identities. It is not a new execution result.
Package identity does not pin floating subprocess resolutions or unversioned model downloads. Preserve that limitation when carrying results forward.

If behavior or relevant inputs change, repeat the installed public `scribe.transcribe_audio` smoke with permitted synthetic audio, default diarization and `include_embeddings=True`.
Require attributed segments and computed voiceprints to exercise recognition, diarization and Senko's native/model path.
Use existing caches and offline resolution. Set `SDL_AUDIODRIVER=dummy`, avoid player commands and clean up subprocess descendants.
Record only synthetic segment counts and embedding dimensions, not transcripts or vectors.
A smoke result is neither an accuracy claim nor permission to redistribute the synthetic voice engine's output.

Root, Scribe, evaluation and dynamic subprocess environments are separate resolutions.
The [package inventory](dependency-inventory.json) and [model record](models-and-data.md) define the observed scope, not future resolutions or arbitrary overrides.
