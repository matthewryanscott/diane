# Licensing and redistribution

Diane's first-party code and local contributions use [MIT](../../LICENSE).
Vendored Scribe retains its [upstream MIT license](../../scribe/LICENSE) and [local modification notice](../../scribe/NOTICE).
**MIT does not relicense dependencies, models, datasets, voices or generated content.**

## Scope and decision

The audit covers application baseline `a815bc62f39d666d78338ac93c3c06589d9f9866`, ten original locks and runtime observations dated 2026-10-22.
The publication candidate retains nine locks. It excludes DiariZen, Cohere's optional MLX Silero VAD path, Qwen's Japanese/Korean tokenizer extra, private-input helpers and identity-linked reports.
English Qwen ASR and forced alignment remain. Core meeting behavior, runtime resolution and model-loading defaults remain unchanged.

The intended release is Git source and Diane/Scribe wheels and sdists for an externally installed Apple-silicon macOS runtime.
It is not a third-party wheelhouse, standalone executable, container, model bundle or Linux-support promise.
The pinned core Senko bundle has affirmative code/model grants. Unknown original checkpoint/conversion revisions limit provenance, not the supplied bundle's permission.
No aggregate environment, native binary, model cache, dataset or private-output redistribution is cleared by this audit.

## Records

- [Scribe provenance](scribe-provenance.md): exact upstream identity and first-party grant.
- [Dependencies](dependencies.md) and [package inventory](dependency-inventory.json): exact versions, source identities, scoped findings and obligations.
- [Models, data and tools](models-and-data.md): independent asset terms, exclusions and operator gates.
- [Artifact checks](artifacts.md): package boundaries and release verification protocol, not a claim that checks passed.

## Distribution boundaries

| Artifact | Included legal material | Excluded payload |
| --- | --- | --- |
| Diane wheel | Root `LICENSE` and `THIRD_PARTY_NOTICES` | Scribe implementation, dependency libraries, evaluation code, models and data |
| Scribe wheel | `scribe/LICENSE` and `scribe/NOTICE` | Senko implementation, subprocess dependencies, models and data |
| Publication source / sdists | Applicable root and Scribe legal files, source and concise provenance | Recordings, transcripts, voiceprints, raw evaluation runs, caches, private-input helpers/reports, external checkouts and installed binaries |

Diane wheel metadata names `scribe` without uv's local source mapping. Install the audited companion wheel, not an unrelated registry namesake.
Retain the four legal files unchanged when redistributing the corresponding source/artifacts.
External dependencies keep their own notices in any distribution that actually includes them.

## Re-audit and release gates

A classifier, SPDX label, nearby release or permissive library license does not prove an exact asset's grant.
Unresolved optional/Linux findings remain scoped holds, not missing core permissions or automatic barriers to first-party macOS source/wheel release.
Before copying an environment or asset, identify its exact bytes and satisfy all applicable notices, copyleft, vendor and access conditions.
Re-audit after lock/source changes, subprocess re-resolution, model-cache refreshes, backend/input changes or a wider distribution boundary.
Do not download large weights merely to inspect license terms.

Recordings, transcripts, voiceprint vectors, speaker labels and identity-linked logs need separate input rights, authorization and disclosure review.
An operator-confirmed name does not establish participant consent. Keep private material out of releases.
This is an engineering record, not legal advice, an ownership warranty or blanket release clearance.
