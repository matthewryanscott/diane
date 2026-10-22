# Nemotron-3-Diarization vs senko

This historical experiment compares Nemotron-3-Diarization with senko on eight public AMI meetings.
Nemotron has 100M parameters, up to eight speakers, overlap-aware output and arrival-order speaker channels.
Two local ports ran: mlx-audio (`nemotron3-mlx`) and FluidAudio's CoreML Swift CLI (`nemotron3-fluidaudio`).
Their pooled AMI DER differed by less than 0.25 points in these runs.
The model-card comparison uses only half its test set, so it is not an exact reproduction.
See [the evaluation index](../README.md) for protocol, reproduction limits and the pending publication gate.

## Method

The configuration of each port is as follows:

- **nemotron3-mlx**: `mlx-audio==0.5.6` (released 2026-09-24) in the separate uv project `eval/envs/nemotron-mlx/`, with weights `mlx-community/Nemotron-3-Diarization` (bf16, converted from `nvidia/Nemotron-3-Diarization` revision `723e19c`). The runner is `eval/run_nemotron_mlx.py`, and the worker is `eval/nemotron_mlx_worker.py`. It uses the card's offline preset (chunk 340, right context 40, FIFO 40, speaker cache 264, update period 300; 30.4 s input buffer).
- **nemotron3-fluidaudio**: a FluidAudio checkout at `3fc97306` (v0.17.2, which includes PR #883 and the follow-up fix #952) in `eval/envs/fluidaudio/`, built with `swift build -c release --product fluidaudiocli`. It runs `fluidaudiocli nemotron3-diarize --variant offline --compute-units all` with the bundles from `FluidInference/nemotron-3-diarization-coreml`. The runner is `eval/run_nemotron_fluidaudio.py`.
- Both ports dump [T, 8] probabilities at 10 ms, and `eval/nemotron_common.py` binarizes both in the same way: threshold 0.5, no padding, no minimum duration. This is NeMo's default when no post-processing YAML is given, which is the card protocol. Nemotron takes no speaker-count hint, and no speaker cap was set other than the model's built-in 8.
- One process runs per file, measured with `measure.py` on the model process itself, with model load included. Other agents were running ASR models at the same time (load average about 8 to 12), so all runtimes are provisional.
- The AMI scoring is `score.py diar` and `score.py combo --asr parakeet-v3`. `eval/nemotron_fa_score.py` also scores against the forced-alignment references that the NVIDIA card uses (see the next section).

Outputs go to `eval/runs/diar/<name>/{sdm,ihm}/<id>.{rttm,meta.json,probs.npy,log}`. Neither port exposes speaker embeddings. The MLX worker also saves `<id>.centroids.npy`, which is the probability-weighted mean of the encoder's final hidden state per speaker, together with `<id>.speakers.json`. These encoder-state centroids are not validated identity embeddings.

## Comparison with published numbers

The card scores AMI against the forced-alignment references from `nttcslab-sp/diar-forced-alignment`, which are much tighter than the harness's pyannote `only_words` references. For example, ES2004a has 708 s of speaker time in the forced-alignment reference and 923 s in the harness reference. These references are copied to `eval/data/ami/ref/rttm_fa/`. Collar 0, overlap scored, 8 of the card's 16 test meetings:

| run | SDM DER c0 | IHM (MHM) DER c0 | card (16 meetings) |
|---|---|---|---|
| nemotron3-mlx | 11.70 | 9.77 | SDM 11.14, MHM 9.25 |
| nemotron3-fluidaudio | 11.89 | 9.76 | FluidAudio PR: MHM 9.47 |
| senko | 51.49 | 46.50 | — |

Both ports were within about 0.8 DER points of the card's scores, but used eight rather than sixteen meetings.
On IS1009a IHM, their binarized frame probabilities agreed at 99.89% (mean absolute difference 0.004).
These ports used bf16 on MLX and fp16 on CoreML.
Senko's merged segments bridge pauses, contributing about 28% false alarm against these tighter references.

## AMI results (harness references, all 8 meetings, pooled)

| metric | senko SDM | mlx SDM | FA SDM | senko IHM | mlx IHM | FA IHM |
|---|---|---|---|---|---|---|
| DER c0 | 33.02 | **27.11** | 27.24 | 28.00 | 25.80 | **25.75** |
| DER c.25 | 24.20 | 24.62 | 24.74 | **19.29** | 23.77 | 23.74 |
| miss / FA / conf (c0) | 16.26 / 7.71 / 9.04 | 24.52 / 1.56 / 1.03 | 24.52 / 1.54 / 1.17 | 15.75 / 7.70 / 4.55 | 23.61 / 1.19 / 1.00 | 23.56 / 1.19 / 1.01 |
| JER | 41.71 | **30.66** | 30.97 | 35.61 | 30.64 | **30.59** |
| speaker count exact, MAE | 5/8, 0.62 | **7/8, 0.12** | 7/8, 0.12 | 3/8, 0.75 | **7/8, 0.12** | 6/8, 0.25 |
| cpWER word merge | 41.44 | **31.64** | 31.86 | 30.48 | 27.39 | **27.38** |
| cpWER sentence merge | 44.18 | **35.38** | 35.70 | 33.52 | **31.13** | 31.15 |
| tcpWER word merge | 43.75 | **33.31** | 33.60 | 31.95 | 28.66 | **28.65** |

Speaker confusion falls from 9.0% (SDM) and 4.6% (IHM) to about 1%. With word-level merge, cpWER improves by 9.8 points on SDM and 3.1 points on IHM. EN2002a SDM, senko's worst case (2 speakers found, cpWER 78%), is now correct at 4 speakers. The only speaker-count error on AMI is ES2004b, where both ports find 5 speakers. FluidAudio also finds 5 speakers on EN2002b IHM, where the probabilities differ slightly between fp16 and bf16.

Against the harness references, Nemotron's DER is almost all missed speech, because it marks the pauses inside the reference's word-run segments as silence. For the same reason, collar 0.25 helps senko by 9 points and Nemotron by only 2, so senko wins DER c.25 on IHM. This is an artifact of the reference style: against the forced-alignment references, Nemotron's miss is 5 to 6% and senko's false alarm is 28%. scribe's word-level merge gives a word that overlaps no speaker the previous word's speaker, so these gaps do not hurt attribution. The cpWER gain shows this.

## Runtime and memory

Runtime is process wall time, including model load, measured per file under contention from other agents.

| run | AMI SDM total s (RTFx) | AMI IHM total s (RTFx) | AMI peak MB |
|---|---|---|---|
| senko | 102.6 (136x) | 131.7 (106x) | 769–1303 |
| nemotron3-mlx | 55.8 (250x) | 59.2 (235x) | 1391–1733 |
| nemotron3-fluidaudio | 54.9 (254x) | 54.4 (256x) | **293–508** |

Model load took 1–6 s for MLX and 2–4 s for FluidAudio.
AMI in-process inference totaled 42 s for MLX (330x) and 28 s for FluidAudio (500x).
Senko's own inference totaled 31–39 s. Import and load overhead explain much of the wall-time difference.
The external 904x FluidAudio figure used inference-only timing on a different idle machine. It is not directly comparable.

## Port maturity

The evaluated mlx-audio 0.5.6 port installed from PyPI without local changes.
`load(..., strict=True)` and `generate_stream` loaded weights of about 400 MB.
It required a separate environment with `transformers>=5.14` and exposed no speaker embeddings.
Current scribe uses an isolated subprocess for this port, not in-process integration.

The FluidAudio port works, but it is rougher around the edges:

- PR #883 merged on 2026-09-23. A crash fix for mismatched output backings (#952, M3 Max on ANE and M5 on macOS 27) followed the next day, and the issue (#951) is still open.
- On this machine, SwiftPM's download of the `NemoTextProcessing.xcframework` binary artifact hung twice, although a direct `curl` of the same URL took 1 s. I downloaded the zip, checked it against the manifest's SHA-256, unzipped it to `eval/envs/fluidaudio/LocalArtifacts/`, and changed `url:`/`checksum:` to `path:` in both `Package.swift` manifests. This is a local workaround and is not committed.
- The offline preset logs `ANECCompile() FAILED` and then runs on the GPU. This is the documented "GPU-only" behaviour.
- The CLI's own RTTM applies a 0.2 s minimum segment. We re-binarized from `--dump-preds` so that both ports get the same post-processing, and the CLI's RTTM is kept as `<id>.cli.rttm`.
- Integration from Python means a subprocess call to a Swift binary.

## Caveats

- AMI is only 8 of the 16 test meetings. The harness references penalize tight speech boundaries, so read DER together with the forced-alignment table and cpWER.
- Runtimes were measured under contention from other agents' ASR jobs and were run once each; they do not establish an idle-machine speed ranking.
- Only the offline preset was evaluated. The lower-latency streaming presets exist in both ports, and the card and the FluidAudio PR show +0.2 to +1.7 DER for them.

## Historical conclusion

The AMI results supported choosing Nemotron for scribe's default diarizer, which now uses the MLX port.
With the 120/15 s parakeet baseline, it lowered cpWER by 9.8 points on SDM and 3.1 on IHM.
These results do not establish a universal model ranking.
FluidAudio used less memory in these runs, but required a Swift build and subprocess integration.
Cross-recording naming needs a separate speaker-embedding model because Nemotron exposes no identity embeddings.

## AMI commands and setup

From `eval/`, restore and sync `envs/nemotron-mlx/` before the MLX run.
For FluidAudio, restore the checkout at `envs/fluidaudio/` to `3fc97306` and build its release CLI first.
The build command is `swift build -c release --product fluidaudiocli` inside that checkout.

```sh
uv run python prepare_ami.py
uv run python run_nemotron_mlx.py --preset offline --meetings all
uv run python run_nemotron_fluidaudio.py --variant offline --compute-units all --meetings all
uv run python score.py diar --name nemotron3-mlx --meetings all
uv run python score.py combo --asr parakeet-v3 --diar nemotron3-mlx --merge word --meetings all
```

The combo command needs the `parakeet-v3` baseline outputs described in [the index](../README.md).
`prepare_ami.py` does not fetch the forced-alignment references.
For that comparison, obtain `AMI/test/*.rttm` from `nttcslab-sp/diar-forced-alignment` at `9527b7c`.
Place the eight AMI references in `data/ami/ref/rttm_fa/`, then run:

```sh
uv run python nemotron_fa_score.py --names nemotron3-mlx,nemotron3-fluidaudio,senko --variants sdm,ihm
```

That scorer requires all eight meetings for every requested system and variant. It has no subset option.
