# DiariZen vs Nemotron and senko

This historical experiment tested DiariZen (`BUT-FIT/diarizen-wavlm-large-s80-md-v2`) on partial AMI SDM subsets.
The segmentation checkpoint uses CC-BY-NC-4.0 and its publisher acknowledges an unresolved training-license conflict.
The runnable path and environment are excluded from this source candidate. The independent PLDA files have a separate CC BY 4.0 grant.
This report retains aggregate observations, not permission or instructions to restore that model.
See the [exclusion evidence and rights-holder gate](../../docs/licensing/models-and-data.md).
Some tested settings lowered DER against the harness references, but did not establish a speaker-attribution advantage over Nemotron.
Measured default CPU speed was 0.5–0.8x real time. Step 0.25 reached 1.3x. MPS tests did not improve end-to-end performance.
See [the evaluation index](../README.md) for protocol, reproduction limits and the pending publication gate.

## Setup

- **Historical code**: `BUTSpeechFIT/DiariZen` at `844f5555b0a98acd0931511fc641a8c5b8ba92c7` (2026-08-04). It included its own pyannote-audio 3.1 fork. The archived Python 3.11 environment installed both as editable dependencies with `torch==2.5.1`, `torchaudio==2.5.1` and `numpy==1.26.4`. The upstream pin was torch 2.1.1 with CUDA. torch 2.5.1 ran without code changes.
- **Historical weights**: the experiment downloaded `BUT-FIT/diarizen-wavlm-large-s80-md-v2` (265 MB, with VBx PLDA files) and `pyannote/wespeaker-voxceleb-resnet34-LM` (25 MB). Both were ungated then. Access was not license clearance.
- **Historical runner**: the removed runner used one worker process per meeting, measured with `measure.py`. It called `DiariZenPipeline.from_pretrained(...)`, then `pipeline(wav)`, recording segmentation, embedding and clustering times. Historical options selected device, step, batch size and speaker-count hints. No speaker-count hints were used in these runs.
- **Historical settings**: the model's `config.toml` used 16 s windows, step 0.1 (a 1.6 s hop), median filtering, VBx (AHC threshold 0.6, Fa 0.07, Fb 0.8, LDA 128), and 1 to 20 speakers. Two variants changed only the step:
  - `diarizen`: default step 0.1.
  - `diarizen-s025`: step 0.25 (a 4 s hop).
  - `diarizen-fast`: step 0.5 (an 8 s hop).
- Segmentation and embeddings used batch size 4. The removed worker exposed batch size as a memory setting, not an accuracy parameter. Batch size 32 exceeded 17 GB on MPS and the process was killed.
- **Device**: CPU, 4 torch threads (the torch default). See the MPS section for why.

Coverage was incomplete: no IHM run, no private proxy, and only the SDM subsets listed below.

| run | step | meetings (SDM) | compute s |
|---|---|---|---|
| diarizen-fast | 0.5 | quick 4 (5536 s of audio) | 1352 |
| diarizen-s025 | 0.25 | ES2004a, IS1009a | 1457 |
| diarizen | 0.1 (default) | IS1009a | 1652 |

## Results

### Two meetings (ES2004a, IS1009a SDM), all settings

| metric | senko | nemotron3-mlx | diarizen-fast (0.5) | diarizen-s025 (0.25) |
|---|---|---|---|---|
| DER c0 | 31.25 | 28.57 | 21.29 | **17.15** |
| DER c.25 | 21.68 | 25.55 | 14.68 | **11.03** |
| miss / FA / conf (c0) | 17.74 / 7.94 / 5.57 | 23.61 / 2.71 / 2.25 | 7.68 / 6.35 / 7.26 | 7.23 / 6.47 / 3.45 |
| JER | 39.10 | 33.75 | 35.27 | **25.07** |
| speaker count exact | 1/2 | 2/2 | 2/2 | 2/2 |
| cpWER word merge (parakeet-v3) | 33.12 | 31.37 | 36.15 | **30.86** |
| tcpWER word merge | 35.16 | 34.64 | 40.21 | **33.43** |
| DER c0, forced-alignment refs | 46.75 | **15.47** | 37.31 | 33.31 |
| runtime s (RTFx) | 19.9 (95x) | **15.7 (120x)** | 360 (5.2x) | 1457 (1.3x) |

Per meeting, IS1009a, all three DiariZen steps:

| run | DER c0 | conf | spk (ref 4) | cpWER | runtime s |
|---|---|---|---|---|---|
| senko | 29.38 | 7.46 | 5 | 31.11 | 9 |
| nemotron3-mlx | 21.92 | 1.56 | 4 | **27.99** | 6 |
| diarizen-fast (0.5) | 23.11 | 10.23 | 4 | 37.06 | 133 |
| diarizen-s025 (0.25) | **16.54** | 3.92 | 4 | 28.19 | 728 |
| diarizen (0.1, default) | 17.11 | 4.82 | 5 (a 19 s extra) | 28.86 | 1652 |

On ES2004a, step 0.25 reached cpWER 32.80 versus Nemotron's 33.84.
Only IS1009a tested both step 0.25 and the default. That one-meeting result does not establish equivalent accuracy.
Step 0.5 raised confusion and cpWER on the two-meeting comparison.

### Quick SDM (4 meetings), diarizen-fast only

| metric | senko | nemotron3-mlx | diarizen-fast |
|---|---|---|---|
| DER c0 | 44.26 | 31.46 | **20.72** |
| DER c.25 | 35.78 | 29.15 | **15.14** |
| miss / FA / conf | 20.10 / 8.19 / 15.97 | 28.42 / 1.61 / 1.43 | 9.63 / 4.85 / 6.24 |
| speaker count exact, MAE | 1/4, 1.25 | **4/4, 0.00** | 3/4, 0.25 (TS3003a: 3) |
| cpWER word merge | 58.67 | **38.82** | 43.16 |
| DER c0, forced-alignment refs | 64.84 | **15.12** | 40.80 |
| runtime s (RTFx) | 45.0 (123x) | 26.2 (212x) | 1352 (4.1x) |

diarizen-fast fixes senko's EN2002a failure (it finds 4 speakers; cpWER 47.45 against senko's 78.13), but it is worse than Nemotron on every meeting's cpWER.

### Why the DER and cpWER rankings disagree

Reference style changes the DER ranking.
DiariZen fills short within-turn pauses, like the harness's `only_words` reference. Its miss rate was 4–10% versus Nemotron's 24–28%.
Against tighter forced-alignment references, DiariZen reached DER 33–41% (25–29% false alarm), versus Nemotron's roughly 15%.
Speaker confusion was 3.5% for step 0.25 versus 2.3% for Nemotron on the two-meeting subset.
Their cpWER values were close: 30.86 versus 31.37. This does not establish a meaningful attribution advantage.

The published 13.9 DER uses all sixteen AMI test meetings at the default step.
Our default run covered only IS1009a, with DER 17.11. It is not a reproduction of that published score.

## Runtime and memory

Runtime is process wall time with model load included (the load takes about 1 s). The machine was shared with other agents' jobs: the load average was 8 at the start and 16 to 21 during the step-0.1 and step-0.25 runs, and the GPU showed 100% device utilization from other jobs.

| run | RTFx (audio s / wall s) | compute per hour of audio | segmentation / embeddings share | peak footprint MB |
|---|---|---|---|---|
| diarizen (0.1), IS1009a, load ~20 | 0.51x | ~2.0 h | 48% / 52% | 4682 |
| diarizen (0.1), 120 s clip, load ~8 | 0.78x | ~1.3 h | 56% / 44% | 7983 (batch 8) |
| diarizen-s025 | 1.3x | ~46 min | 50% / 50% | 4614–4664 |
| diarizen-fast | 4.1x (3.0x–6.3x) | ~15 min | 51% / 49% | 4274–4727 |
| nemotron3-mlx (for comparison) | 212x | ~17 s | — | 1391–1733 |
| senko | 123x | ~30 s | — | 769–1194 |

The cost is structural. The 16 s window slides by 1.6 s, so each second of audio passes through WavLM-Large 10 times. Each (window, local speaker) pair also gets its own ResNet34 embedding. That is about 2250 WavLM windows and up to about 9000 embeddings per hour of audio. One 16 s window takes 0.54 s on CPU. The memory footprint is about 4.5 GB at batch 4, even for 30 s of audio. It is 3 times Nemotron's and 5 times senko's.

The optional private speaker-count check was skipped because of runtime.
No private-dataset accuracy or long-file result follows from these AMI measurements.

## MPS

MPS works with a one-line change. `DiariZenPipeline` hardcodes `cuda:0` or `cpu`, and the worker then calls pyannote's `pipeline.to(torch.device("mps"))`. On a 30 s clip the output was the same as on CPU (the same speakers and durations), but MPS was slower, and it used more memory:

| test | CPU | MPS fp32 | MPS fp16 |
|---|---|---|---|
| segmentation forward, 1 × 16 s | 0.54–0.58 s | 0.68–0.85 s | — |
| segmentation forward, 4 × 16 s | 3.3 s | 3.2 s | 2.0 s (after warm-up) |
| 30 s clip, full pipeline (batch 4) | 17.0 s, 4.6 GB | 29.9 s, 7.6 GB | — |
| 120 s clip, batch 32 | — | killed after 10 min at 17 GB | — |

The profile shows most of the MPS time as device-sync waits (`aten::copy_`). The GPU was fully used by other agents at the time, so an idle-GPU number could be better. Even a 5x MPS speedup would still leave DiariZen 20 to 40 times slower than Nemotron, with a 3 to 7 GB footprint. fp16 would also mean patching the model and the pipeline. MPS was not pursued further.

## Historical conclusion

The experiment did not adopt DiariZen for scribe.
Step 0.25 reached cpWER 30.86 versus Nemotron's 31.37 on two SDM meetings, an unresolved small difference.
Its default and step-0.25 settings cost about 46 minutes to two hours of CPU per audio hour under contention.
Step 0.5 was faster, but had cpWER 43.16 versus Nemotron's 38.82 on the quick subset.
It counted speakers exactly on 3/4 quick meetings, not 4/4.
The non-commercial weights also limit reuse. These observations do not rank untested devices, meetings or settings.

## Caveats

- Only SDM was run. The default step ran on 1 meeting, step 0.25 on 2 meetings, and step 0.5 on the 4 quick meetings. The two-meeting cpWER comparison (30.86 vs 31.37) is within noise. ES2004a and IS1009a are also the easy meetings, where senko is at 33.12.
- All runtimes came from one pass on a heavily loaded shared machine. Treat speed ratios as provisional, not idle-device measurements.
- The removed runner supported speaker-count hints, but this experiment did not test them.
- Run outputs are in `eval/runs/diar/{diarizen,diarizen-s025,diarizen-fast}/sdm/`: `<M>.rttm`, `<M>.meta.json` (with phase timings in `diarizen`), `<M>.diarizen.json` and `<M>.log`. The forced-alignment DER for these subsets was computed with the same logic as `eval/nemotron_fa_score.py`, which only scores all 8 meetings.

## Historical reproduction limits

The runner, worker and environment are excluded from this source candidate.
Do not restore or run the archived model without resolving the [rights-holder gates](../../docs/licensing/models-and-data.md).
This report supplies no runnable DiariZen reproduction command.
Historical scoring also depended on original ignored outputs and matching Parakeet baseline outputs, which are not distributed here.
`nemotron_fa_score.py` requires all eight meetings and has no subset option.
The partial forced-alignment tables used its scoring logic manually, not a checked-in subset command.
The isolated MPS forward-pass and fp16 timings also lack a dedicated reproduction command.
