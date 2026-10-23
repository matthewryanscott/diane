# Models, data, outputs and external tools

## Scope and core conclusion

The audit covers application baseline `a815bc62f39d666d78338ac93c3c06589d9f9866` and primary-source observations dated 2026-10-22.
Diane/Scribe source and wheels contain no model weights or AMI audio. External dependencies and downloads retain separate terms.
The pinned Senko dependency **does bundle models and native libraries**. Selecting Nemotron still uses Senko to compute voiceprints.
The supplied core bundle has affirmative permission grants. Unknown original checkpoint/conversion revisions limit provenance, not its supplied permission.
No model bytes, private recordings, transcripts or identity-linked outputs were inspected for the model audit.

Model links below identify exact audited public repository revisions, not runtime pins or weight checksums.
Only the Parakeet-v3 and MLX Nemotron revisions also had matching local cache refs/snapshot directory names.
Those observations prove neither file completeness nor which historical command used them.
The loaders generally omit model revisions. `parakeet-mlx>=0.4,<1` and `mlx-audio==0.5.6` run in separate `uvx` environments.
The root lock does not govern those environments or their model downloads.

## Core model assets

| Model and audited identity | Terms and scoped finding |
| --- | --- |
| [`mlx-community/parakeet-tdt-0.6b-v3`](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3/blob/ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15/README.md), upstream [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/blob/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/README.md) | **CC BY 4.0**, including commercial use. Preserve attribution, source/license links and modification information when sharing. Product selection follows the Parakeet package default, not an explicit model pin. |
| [`mlx-community/Nemotron-3-Diarization`](https://huggingface.co/mlx-community/Nemotron-3-Diarization/blob/59ed2dbfc1346dcea9d423c71306a3a2499c568f/README.md), upstream [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/f667ed73aee57d40cc39428eb768b4fd87a0a29e/README.md) | **OpenMDW-1.1**, including commercial use. Distributed Model Materials need the agreement and origin notices. The agreement includes litigation termination but imposes no output-use/sharing restrictions. Other rights and consent remain separate. Historical upstream report revision resolves to `723e19c601d99b7e58fba6a14e32153e0afe48d9`, not a converted-model pin. |
| Senko CAM++ `speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt`, `camplusplus_batch16.mlpackage`, `camplusplus_traced_cuda_optimized.pt` | **Apache-2.0** under Senko's component notice. Origin: ModelScope `iic/speech_campplus_sv_zh_en_16k-common_advanced`, plus Senko CoreML/CUDA conversions. Original upstream checkpoint revision is unknown. Preserve component notices and conversion provenance. |
| Senko `pyannote_segmentation_3.0/pytorch_model.bin`, `pyannote_segmentation.mlmodelc` | **MIT**, copyright CNRS, under Senko's supplied notice. macOS initializes CoreML VAD even for embedding use. Upstream [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0/tree/e66f3d3b9eb0873085418a7b813d3b369bf160bb) declares MIT and gates direct access. Its current revision does not identify Senko's original conversion. |
| Packaged Python Silero JIT/ONNX assets | Root `silero-vad==6.2.1` supplies Senko's non-Darwin fallback under [MIT](https://github.com/snakers4/silero-vad/blob/v6.2.1/LICENSE). Stereo evaluation resolves Silero independently. Its dated R3 resolution selected `silero-vad==6.2.3`, `torch==2.11.0`, `numpy==2.5.4` among 13 distributions. The [v6.2.3 MIT grant](https://github.com/snakers4/silero-vad/blob/5cd7945676eb32225748052e2e6a0580e4686a08/LICENSE) covers packaged models. R3 was metadata-resolved, not installed/executed, and does not pin future runs. |

Senko is pinned to [`narcotic-sh/senko@40250276a691489786bd20bbebffa609dbf5acb0`](https://github.com/narcotic-sh/senko/tree/40250276a691489786bd20bbebffa609dbf5acb0).
Its [model origins](https://github.com/narcotic-sh/senko/blob/40250276a691489786bd20bbebffa609dbf5acb0/models/README.md), [component licenses](https://github.com/narcotic-sh/senko/blob/40250276a691489786bd20bbebffa609dbf5acb0/THIRD_PARTY_LICENSES) and [build rules](https://github.com/narcotic-sh/senko/blob/40250276a691489786bd20bbebffa609dbf5acb0/CMakeLists.txt) cover its supplied bundle and installation of `models/` into its wheel.
The observed native wheel used scikit-build-core 1.1.1 and tag `cp313-cp313-macosx_26_0_arm64`. This does not attest its historical compiler/SDK.

Supporting CoreML origin [`FluidInference/speaker-diarization-coreml@7eab74f3463c736418d0eb10296261ae581146c9`](https://huggingface.co/FluidInference/speaker-diarization-coreml/blob/7eab74f3463c736418d0eb10296261ae581146c9/NOTICE.md) distinguishes legacy MIT segmentation from current Community-1 CC BY 4.0 material.
Its legacy notice says 2023 CNRS. Senko's supplied notice says 2020 CNRS. Both affirm MIT permission.
Preserve supplied notices and reconcile provenance before a combined redistribution. The original legacy conversion scripts were not preserved.

## Evaluation and historical selections

These are named model identities, not clearance for arbitrary `--model`, `--aligner` or local-directory overrides.
Apply the same CC BY, Apache and OpenMDW redistribution duties described above.

| Model and audited revision | Terms, selection and remaining limit |
| --- | --- |
| [`mlx-community/parakeet-tdt-0.6b-v2`](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2/blob/8ae155301e23d820d82aa60d24817c900e69e487/README.md), upstream [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/blob/ae9ad07059c7c739ffaf932226a8fe64ae2620b0/README.md) | **CC BY 4.0**. Historical experiment/override option. Historical converted weight revision remains unknown. |
| [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b/blob/de575db64086f84fdc79da4932d1076e965bc546/README.md) | **Apache-2.0**. Granite worker loads an overridable model without revision. Preserve notices and identify modified/converted files. |
| [`littoralai/cohere-transcribe-mlx-fp16`](https://huggingface.co/littoralai/cohere-transcribe-mlx-fp16/blob/6a66668d9b9b26646fce93d70412ee33188b17e0/LICENSE), upstream [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/tree/d552dadc2922ba6f6c1f994f8db904d52dcaf942) | **Apache-2.0**. Conversion identifies unchanged Cohere custom code and fp16 conversion. Direct upstream access is gated. An ungated conversion does not establish acceptance of direct-upstream conditions or a supplied local directory's identity. |
| [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/7278e1e70fe206f11671096ffdd38061171dd6e5/README.md) | **Apache-2.0**, supported by the [Qwen family license](https://github.com/QwenLM/Qwen3-ASR/blob/main/LICENSE). Overridable, without revision. |
| [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md) | **Apache-2.0**. Historical experiment through the same loader. Historical downloaded revision remains unknown. |
| [`Qwen/Qwen3-ForcedAligner-0.6B`](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/blob/c7cbfc2048c462b0d63a45797104fc9db3ad62b7/README.md) | **Apache-2.0**. Separate weight dependency used by Qwen/alignment workers, without revision. English ASR/alignment remain after the tokenizer-extra exclusion. |
| [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1/tree/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee) | Publisher declares **CC BY 4.0** for the pipeline. Loader omits revision. Independent embedding/PLDA notices remain uninspected because raw access returned 401. Direct access requires human review of profile disclosure, company/university/use-case questions and product/service-email agreement. |
| [`FluidInference/nemotron-3-diarization-coreml`](https://huggingface.co/FluidInference/nemotron-3-diarization-coreml/blob/25a90f97f254428d4b30374b76af9c74fdee8327/README.md) | **OpenMDW-1.1**. Evaluation selects `offline`. FluidAudio's upstream loader uses `main`, not this weight pin. Include learned-silence/projection assets in any redistribution inventory. The card's abbreviated `.nemo` hash is not a complete artifact identity. |
| [`pyannote/wespeaker-voxceleb-resnet34-LM`](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/blob/837717ddb9ff5507820346191109dc79c958d614/README.md) | **CC BY 4.0**. Historical DiariZen embedding dependency. Actual historical revision is unknown. The declaration does not independently clear VoxCeleb training-data rights. |
| [`Wespeaker/wespeaker-voxceleb-resnet34-LM`](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/blob/f0c48c298fd835726c27956a5d617bad7115627e/README.md) | **CC BY 4.0**. Distinct repository. Excluded private-input experiment used an unidentified local ONNX conversion. Restore only with origin, revision/hash, attribution and input/privacy clearance. |

### Excluded paths and reinstatement gates

- **DiariZen:** runner, worker and environment/lock are excluded. [`BUT-FIT/diarizen-wavlm-large-s80-md-v2@027b3e221b9d81dce2be794084f1dbd3ba2da403`](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2/blob/027b3e221b9d81dce2be794084f1dbd3ba2da403/README.md) separates MIT code from **CC BY-NC 4.0 checkpoint weights**. Code revision `BUTSpeechFIT/DiariZen@844f5555b0a98acd0931511fc641a8c5b8ba92c7` includes the modified pyannote subtree. Its `MODEL_LICENSE` acknowledges unresolved CC BY-SA/non-commercial training-data conflicts. The checkpoint is neither commercially cleared nor unconditionally cleared for research redistribution. Reinstatement needs rights-holder/legal resolution for the actual checkpoint. Its separate [`plda/LICENSE`](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2/blob/027b3e221b9d81dce2be794084f1dbd3ba2da403/plda/LICENSE) grants **CC BY 4.0**, including commercial use, for `plda.npz` and `xvec_transform.npz`, copyright BUT Speech@FIT.
- **MLX Silero:** Cohere's VAD flags/branch are excluded and its worker uses `vad=False`. [`mlx-community/silero-vad@7bc17f22d3c0451bd3a6cd71e759b009271ff49a`](https://huggingface.co/mlx-community/silero-vad/blob/7bc17f22d3c0451bd3a6cd71e759b009271ff49a/README.md) has no supplied license declaration/file. Its upstream [`onnx-community/silero-vad@e71cae966052b992a7eca6b17738916ce0eca4ec`](https://huggingface.co/onnx-community/silero-vad/blob/e71cae966052b992a7eca6b17738916ce0eca4ec/LICENSE) supplies MIT, but conversion permission/provenance remains unresolved. Restore only with a grant covering the converted assets. This is not packaged Python Silero or a missing Senko core grant.
- **Qwen tokenizers:** only the Japanese/Korean extra is excluded. Exact-release SoyNLP/DyNet38 grant/provenance holds remain historical in [Dependencies](dependencies.md).

`aufklarer/Qwen3-ASR-0.6B-MLX-4bit` was reported cached but not used. Official Qwen findings do not clear it.
`gemma4:e4b` was a default in the retired, unused server loader, not an executed model dependency. No product request consumer was established.
The audited [Ollama manifest](https://registry.ollama.ai/v2/library/gemma4/manifests/e4b) SHA-256 is `dc35e8d9c6061baa6f0fa870975ab6932e2542b579b13ea0f199fa4bb7300c9c`.
Its [Apache-2.0 license layer](https://registry.ollama.ai/v2/library/gemma4/blobs/sha256:7339fa418c9ad3e8e12e74ad0fd26a9cc4be8703f9c110728a992b193be85cb2) digest is `sha256:7339fa418c9ad3e8e12e74ad0fd26a9cc4be8703f9c110728a992b193be85cb2`.
The mutable tag does not identify an operator's installed model.

## Evaluation data

[AMI's publisher](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) licenses corpus audio and manual annotations under **CC BY 4.0**, not research-only terms.
Preparation downloads corpus-mirror audio, `ami_public_manual_1.6.2.zip` annotations and moving-`main` pyannote RTTM/UEM references.
The eight-meeting subset is ES2004a/b, IS1009a/b, TS3003a/b and EN2002a/b. `sdm` selects Array1-01 and `ihm` selects Mix-Headset.
Converted WAV, channel audio, SegLST, RTTM and metadata remain dataset derivatives. Historical audio/archive hashes are unknown. HTTP annotation URLs do not attest obtained bytes.

- [Pyannote AMI setup at `67c2d539286e89f68952d5dcf83912bd9f01dfae`](https://github.com/pyannote/AMI-diarization-setup/blob/67c2d539286e89f68952d5dcf83912bd9f01dfae/LICENSE) supplies **Apache-2.0**, alongside underlying AMI CC BY 4.0 provenance. The preparation URL does not pin that audited revision.
- [Forced-alignment references at `9527b7c64846fb38316a610f32e9d3466bd6d8b7`](https://github.com/nttcslab-sp/diar-forced-alignment/blob/9527b7c64846fb38316a610f32e9d3466bd6d8b7/LICENSE) assign **CC BY 4.0 to AMI** and CC BY-SA 4.0 to AliMeeting. This harness selects AMI, not AliMeeting. Historical copied-file hashes remain unknown.
- Digital silence/noise probes name no separately downloaded corpus. AMI edge excerpts still inherit AMI terms.
- Caller-supplied/private recordings have **no established public reuse license or participant consent**. Private-input helpers and identity-linked reports are excluded.

Before sharing audio, annotations, protected adaptations or transcript examples, retain creator/source/license/disclaimer notices and identify changes.
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode) permits commercial use but excludes privacy/publicity rights from its grant.
A benchmark score does not license recordings or quoted text. Aggregate metrics need provenance/method notes. Examples and per-recording outputs need separate clearance.

## External executables and native redistribution

These tools are externally installed, not bundled into Diane/Scribe wheels.
Invoking them does not relicense first-party source. Shipping their binaries creates additional obligations.

| Tool and audited identity | Terms and redistribution gate |
| --- | --- |
| FFmpeg **9.0.2**, Homebrew **9.0.2_1** | Observed `--enable-gpl --enable-version3`, libx264/libx265 and `ffmpeg -L`: **GPL-3.0-or-later**, not LGPL-only. [FFmpeg legal guidance](https://ffmpeg.org/legal.html) explains component-dependent licensing. Bundling needs corresponding source/build compliance and actual linked-component notices. The observation does not independently identify ffprobe/ffplay binaries. |
| uv **0.13.5**, upstream commit `46776a95d6686ac0160c10611327ce491e6b202f` | [MIT](https://github.com/astral-sh/uv/blob/0.13.5/LICENSE-MIT) or [Apache-2.0](https://github.com/astral-sh/uv/blob/0.13.5/LICENSE-APACHE). Preserve the selected grant's notices. Upstream release identity does not attest Homebrew patches, downloaded Python or resolved environments. |
| FluidAudio **v0.17.2**, [`3fc97306612d119337c487ce789c6cd631a7876b`](https://github.com/FluidInference/FluidAudio/tree/3fc97306612d119337c487ce789c6cd631a7876b) | **Apache-2.0**. The local CLI's exact build/checkout was not attested. Nemotron uses moving `main`; registry overrides and existing caches can change assets. |
| FluidAudio `NemoTextProcessing.xcframework.zip` **v0.3.1**, SHA-256 `5fa8c10d4ec26c1bb2413125f351a7222a4c68a23b74476680fbada7e26fc6aa` | Manifests select v0.3.1 while bundled notice says v0.3.0. Reconcile actual binary contents and this mismatch before compiled-CLI redistribution. [v0.3.1 notices](https://github.com/FluidInference/text-processing-rs/blob/v0.3.1/THIRD-PARTY-LICENSES.md) cover Apache NeMo grammars/fixtures at `1f1263579fe57ba7ed783cad3dddee710fcc5064` and MIT/Apache Rust components. Toolchain/trait selection was not inspected. |
| Swift / Apple platform | [Swift](https://github.com/swiftlang/swift/blob/main/LICENSE.txt) uses Apache-2.0 with Runtime Library Exception. This does not license Apple SDKs, CoreML, Accelerate, Foundation or `/usr/bin/time` for redistribution. No Apple SDK/system binary is established as bundled. |

Preserve FluidAudio's full applicable notice bundle, including fastcluster's BSD-style two-condition attribution, Apache-2.0 VBx and relevant G2P notices.
The selected CLI command alone does not prove which resources a binary includes.
Senko native libraries and model caches also need artifact-level inventories when bundled. See [Dependencies](dependencies.md) for native-build and platform restrictions.

## Operator gates and output rights

These are release/operator obligations, not runtime-enforced checks.

1. For mutable models, overrides, local paths, mirrors, editable Senko checkouts or inherited FluidAudio registry/cache settings, establish actual origin, revision/hash, notices and access permission.
2. Before direct gated access, a human must review/accept current terms and authorize the exact profile disclosure. A token or cached copy is not acceptance evidence.
3. Before bundling environments, models, binaries or data, satisfy all included notices, source/build duties, vendor conditions and the scoped holds above.
4. Establish authorization and an applicable lawful basis for recording, diarization, cross-recording enrollment/matching, retention and disclosure.
5. Review each proposed output: transcripts, frontmatter, `.voice.json`, centroids, accepted names, mappings, reference extracts, raw JSON/RTTM/SegLST, arrays and logs.

Operator-confirmed names do not prove consent. Cluster IDs and embeddings can identify or link people. Logs can disclose content or paths.
A software/model license does not automatically license generated transcripts or clear confidential input expression.
OpenMDW leaves outputs unrestricted by that agreement, not by every other right.
Do not publish private audio, transcript excerpts, voiceprints or identity-linked sidecars without separate artifact-specific clearance.
The excluded DiariZen/MLX Silero paths remain reinstatement gates, not current core-runtime blockers.
