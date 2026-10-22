# Dependency licenses

## Scope and conclusion

This is the 2026-10-22 dependency audit, compacted without new clearance claims. It is not a legal opinion.

The [inventory](dependency-inventory.json) is the canonical dependency list. It preserves 315 exact package identities: 272 retained and 43 historical-only. Three separate source components cover DScore, host CMake and the Apple toolchain. The historical baseline covers ten locks at `a815bc62f39d666d78338ac93c3c06589d9f9866`. The audited candidate retains nine after the authorized DiariZen exclusion.

The intended distribution is first-party Git source and Diane/Scribe wheels or sdists, with an Apple-silicon macOS runtime. Dependency declarations do not mean dependency binaries are bundled. This audit does not clear third-party environment/model bundles or promise Linux/CUDA support. A component-specific hold does not prohibit distributing first-party artifacts that do not contain that component.

The root [MIT license](../../LICENSE) covers first-party source and documentation within [THIRD_PARTY_NOTICES](../../THIRD_PARTY_NOTICES). Original [Scribe MIT terms](../../scribe/LICENSE) and its [modification notice](../../scribe/NOTICE) remain separate. The companion Scribe source is not interchangeable with an unrelated registry package named `scribe`. Virtual evaluation manifests are local project metadata, not independent third-party distributions. Eight retained first-party rows received the scoped MIT grant. Historical `eval-diarizen` retains its unresolved baseline status.

Package grants do not authorize independent model weights, training data, gated services or private inputs.

## Reading the inventory

- Each package appears once per exact name/version/source identity. IDs remain unchanged from the audited inventory. Repeated scopes are unioned. Scribe's directory and editable source identities remain distinct.
- `scopes` identifies retained lock, runtime and build scopes. `excluded_scopes` preserves removed occurrences, including shared packages that remain elsewhere. Scope definitions contain lock counts, hashes and Python ranges.
- `project_terms` records the audited grant, using SPDX names where applicable. `license_sources` points to primary public release archives or immutable source. Archive `members` identify legal files inside that artifact. First-party `license_files` point to the actual repository grants.
- `mentioned_terms` indexes legal-text signatures. It is not a conjunctive SPDX expression, a replacement for bundled terms, or proof that every named library is linked.
- `obligations` references the shared catalog. `distribution_assessment`, notes and release gates distinguish permissive conditions, copyleft, vendor agreements and unresolved provenance.
- `conditional_edges` preserves marked, extra and development introductions. A lock record is not proof of activation or installation. Evaluate the interpreter, platform and selected groups against the actual lock.

The compact inventory omits raw METADATA, retrieval logs, capture paths, rendered HTML and repeated legal texts. URLs identify the original primary sources, not missing local captures. Metadata and classifiers alone never verified permission.

The original audit read large-wheel legal members through HTTP Range ZIP access with CRC checks. Expected lock archive hashes did not prove whole-archive verification. Small downloaded sdists were hash-checked. Supplemental native macOS legal payloads were inspected, but not every platform/Python wheel was compared. A different binary artifact can carry different bundled terms.

## Runtime and build boundaries

**R1/R2:** Offline metadata inspection observed 49 Parakeet distributions and 35 MLX Audio distributions, with 56 distinct name/version pairs combined. Parakeet resolved to 0.5.3 under `>=0.4,<1`. MLX Audio was 0.5.6. Exact installed legal payloads and release grants were inspected without importing application code. These observations do not pin future resolutions or prove complete source-to-installed-binary provenance.

**R3:** The stereo runner independently selects Python 3.13, `silero-vad` and `numpy` through `uv run --no-project`. The offline cache lacked compatible Torch. Registry metadata resolved 13 macOS-arm64 identities, including Silero-VAD 6.2.3 and Torch 2.11.0. That environment was not installed or executed. The Silero wheel contains JIT/ONNX assets. Its MIT license matches pinned upstream v6.2.3. This distribution grant is not a training-data/privacy audit or clearance of the separate MLX-community conversion.

**B1:** The explicit packaging environment observed Hatchling 1.32.4, Packaging 26.3, Pathspec 1.1.1, Pluggy 1.7.0, TOMLKit 0.15.2 and Trove Classifiers 2026.9.21.13. The intended artifact build uses that interpreter without build isolation. This dependency audit did not execute the build. The project still declares unversioned Hatchling. Tool use does not impose the tool's license on generated artifacts, but conveying its environment retains its own obligations, including Pathspec's MPL-2.0.

**B2:** Senko pins Git commit `40250276a691489786bd20bbebffa609dbf5acb0` and declares scikit-build-core. No tracked local patch or override is declared. Its cached native wheel records scikit-build-core 1.1.1 and tag `cp313-cp313-macosx_26_0_arm64`. Current backend metadata observed scikit-build-core, Packaging and Pathspec. Its dynamic `ninja>=1.5` requirement resolved to 1.13.2, not an installed or historical-build claim.

Host CMake 4.4.4 met Senko's 3.18 minimum, so the backend requested no CMake package. CMake's BSD grant and bundled cmsys/libuv/Zstandard notices remain separate. The observed compiler was AppleClang 21.0.0, `clang-2100.3.34.2`, from Xcode 27.1 build `27A9275`.

The observed Xcode agreement was EA2002, dated 2026-06-08. Its [official URL](https://www.apple.com/legal/sla/docs/xcode.pdf) is a reference, not a successful network-retrieval claim. Sections 2.2.A and 2.4 address macOS development and agreement-compliant app/library distribution. The toolchain is not bundled. No agreement acceptance or blanket redistribution grant is inferred. The cached wheel does not establish historical Ninja/CMake/compiler versions or the exact SDK subversion. No cold native build or toolchain reconstruction occurred.

## Distribution conditions and holds

**GPL is a distribution condition, not a missing grant.** PrimePy 1.3's implementation matches upstream commit `6b3aca2bd42663c174c69022f966b5fbb5323c70` after newline normalization. That source carries GPL-3.0, unlike current MIT code. Its retained introduction is `pyannote-audio → torch-audiomentations → torch-pitch-shift → primepy`. Conveying a covered combined work requires GPL-compatible terms, Corresponding Source and applicable installation information. Private use and first-party-only distribution are different surfaces.

**Weak copyleft and bundled code retain their terms.** SoXR/libsoxr use LGPL-2.1-or-later. SoundFile's BSD wrapper does not replace libsndfile/codec terms. Certifi uses MPL-2.0, and tqdm has MIT/MPL components. Setuptools vendors LGPL-3.0 autocommand and MPL-2.0 validate-pyproject. Covered-source, modification and applicable relinking/replacement obligations depend on the conveyed artifact.

NumPy/SciPy native notices include GCC runtimes under GPL with GCC Runtime Library Exception 3.1. Qualifying independent modules do not become GPL merely because those notices exist. Preserve runtime terms and assess actual compilation/linkage. PyTorch, ONNX Runtime, gRPC, Pillow, LLVM/llvmlite, Numba, Core ML Tools and TorchCodec also carry separate component notices. Docutils has a public-domain dedication with enumerated BSD/GPL exceptions. Matplotlib/fonttools font terms can include reserved names. Do not flatten these materials to umbrella metadata.

**NVIDIA grants are component-specific.** Review deployment restrictions, authorized redistributables, notices and agreement requirements for the selected binaries. Linux-marked records do not prove a macOS CUDA installation. NCCL 2.30.7 has an actual BSD-3-Clause archive grant despite proprietary metadata. NVTX releases with Apache text also retain those grants. Neither fact clears all NVIDIA components.

**One retained hold concerns Linux/vendor aggregates:** `cuda-toolkit==13.0.3.0` describes a metadata-only wheel with no software and no grant. The [CUDA 13.0.3 EULA](https://docs.nvidia.com/cuda/archive/13.0.3/eula/index.html) does not establish metapackage applicability or unrestricted aggregate redistribution. Obtain vendor clarification before conveying that surface. Individual verified component terms remain applicable. This is not a macOS first-party release hold.

## Authorized exclusions

**DiariZen:** The optional runner, worker, environment and editable overrides were removed. Historical source recovery identifies clean `BUTSpeechFIT/DiariZen@844f5555b0a98acd0931511fc641a8c5b8ba92c7`, not a revision pinned by the editable lock. MIT code attribution remains BUT Speech@FIT. Its modified, CNRS-copyright pyannote-audio subtree is not the same-version PyPI artifact or a separately pinned repository. The README identifies changes to model, inference, getter and diarization modules. Its separate pretrained-weight terms are CC-BY-NC-4.0 and acknowledge training-data compatibility concerns. DScore remains an uninitialized BSD-2-Clause Gitlink at `e02f949ac6592279300a2c33d03daf9e0c12fd27`, not an installed uv dependency.

**Qwen tokenizers:** The final environment uses base `mlx-qwen3-asr==0.4.4`, not `[aligner]`. It removes only optional Japanese/Korean tokenizers. Pinned source `41878a11cf338c59e13edc84bf4cb35f4b0f3ff6` confines those lazy imports to the language-specific paths. English ASR and ForcedAligner remain. No model or alignment smoke run supports a broader claim.

Historical SoyNLP 0.0.493 declares GPL v3 in source metadata, while later same-version LGPL source differs. Historical DyNet38 2.2 wheels omit legal text, and the linked Apache fork does not establish exact source-to-binary provenance or bundled Eigen/backend grants. The inventory preserves the primary references, comparisons and precise gaps. Obtain matching grants/provenance before reintroducing or conveying those tokenizers. Their removal does not clear unrelated models or waive obligations in shared dependencies.
