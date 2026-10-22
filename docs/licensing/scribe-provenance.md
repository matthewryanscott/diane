# Scribe provenance and first-party grant

The 2026-10-22 audit covers application baseline `a815bc62f39d666d78338ac93c3c06589d9f9866`.
Scribe's original Diane import, `e68acaf200b3c866facc866f2a725eb1658ee6ed`, content-matches [Trail of Bits Scribe at `dda3b31a3077ab847437c82c5e3a4730d1327c78`](https://github.com/trailofbits/scribe/tree/dda3b31a3077ab847437c82c5e3a4730d1327c78).
This is byte-level provenance, not a submodule relationship or upstream endorsement.

The imported `scribe.py`, `LICENSE`, `README.md`, tests, `.gitignore` and `.github/dependabot.yml` match upstream.
The imported `pyproject.toml` changes the Python upper bound to `<3.14`. Its lock differs, and `.nwave/des-config.json` is local configuration.

| Identity | SHA-256 |
| --- | --- |
| [Upstream MIT LICENSE](https://github.com/trailofbits/scribe/blob/dda3b31a3077ab847437c82c5e3a4730d1327c78/LICENSE) and retained `scribe/LICENSE` | `691770e6df61be8ff1919093612d99e5b03a9027ff897eaf039357b7e9054122` |
| Original upstream/imported `scribe.py` | `dd8fb34fc97b081b2159e11c4a6decdd2de76490f2a12ff3d506b78766083332` |
| Application-baseline modified `scribe/scribe.py` | `9be3d45968697e99822f6fd0c01e16255c98417805c5eea403060c51f72f0b85` |

The original notice is `Copyright (c) 2026 Trail of Bits, Inc.`
MIT permits modification and redistribution with the copyright and permission notice retained. It requires neither renaming nor publication of changes.
Retain [scribe/LICENSE](../../scribe/LICENSE), [scribe/NOTICE](../../scribe/NOTICE) and the root [THIRD_PARTY_NOTICES](../../THIRD_PARTY_NOTICES).
No separate dependency/model license text is required merely because Scribe references externally installed components.
Redistributing those components creates separate obligations under the [licensing scope](index.md).

## Local contributions

Local changes add Nemotron/mlx-audio diarization, word-level merging, embeddings, recognition windows, audio-preparation ownership, worker lifecycle and error diagnostics.
Supporting tests and documentation also differ. `scribe/NOTICE` distinguishes this modified copy from the upstream release.
Git attributes those contributions to Matthew Scott. The owner-directed first-party license choice is MIT, including local contributions and documentation.
The root [LICENSE](../../LICENSE) names Matthew Scott and does not replace the upstream grant or dependency notices.
Git authorship supports provenance. It does not prove all legal ownership, settle employment assignments or warrant third-party rights.
