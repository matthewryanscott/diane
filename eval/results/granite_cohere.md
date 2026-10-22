# Granite-speech-4.1-2B and Cohere Transcribe

This historical report summarizes runs from 2026-09-24 in `eval/runs/asr/{granite-4.1-2b*,cohere-2b*}`.
Runners are `eval/run_granite.py` and `eval/run_cohere.py`, with `eval/envs/granite-cohere/gc_worker.py`.
`eval/gc_collapse_loops.py` removes repeated loops for a separate scoring comparison.
See [the evaluation index](../README.md) for protocol, reproduction limits and the pending publication gate.
The optional MLX Silero VAD path and flags are excluded because the captured conversion supplies no license grant.
The VAD rows below are historical observations, not supported reproduction commands.
See the [licensing evidence and exclusions](../../docs/licensing/models-and-data.md).

## Result

Neither model improved on per-turn parakeet in the completed AMI quick-SDM comparison.
Granite stopped before broader coverage because of measured speed. Forced alignment was not evaluated.
This is a limited candidate decision, not a universal model ranking.

Quick subset (ES2004a, IS1009a, TS3003a, EN2002a), SDM, pooled:

| run | WER | sub | del | ins | speed (x real time) | peak memory |
|---|---|---|---|---|---|---|
| parakeet-v3 (historical 120/15 s baseline) | 37.06 | 7.78 | 27.71 | 1.57 | 32.9 | 6.0 GB |
| parakeet-v3-turns (per-turn decoding) | 33.85 | 7.71 | 24.40 | 1.74 | 40.5 | 3.3 GB |
| cohere-2b | 43.60 | 8.05 | 32.08 | 3.47 | 19.1 | 6.0 GB |
| cohere-2b-c15 | 45.18 | 8.91 | 30.77 | 5.50 | 10.9 | 5.3 GB |
| cohere-2b-vad | 40.83 | 5.46 | 34.22 | 1.15 | 13.8 | 6.5 GB |
| cohere-2b-vad15 | 42.50 | 7.41 | 33.54 | 1.55 | 13.9 | 6.1 GB |
| granite-4.1-2b (raw) | 43.54 | 15.09 | 24.56 | 3.89 | 1.1 | 9.8 GB |
| granite-4.1-2b (repetition loops removed) | 37.69 | 5.88 | 30.65 | 1.16 | 1.1 | 9.8 GB |

## Notes

- Granite's raw output contains greedy repetition loops (the 15% substitutions); removing them brings it level with parakeet's default, still behind per-turn parakeet, at ~1x real time.
- A later worker patch replaces Granite's positional-attention broadcast with an einsum. The reported improvement was ~9.8 to ~5.8 GB peak memory and ~8x faster encoder execution with identical bf16 output. The table predates that patch and does not measure the patched end-to-end runtime.
- Quick-IHM and noise-input artifacts were not assessed in this report. They support no result here.
- Runtimes came from a shared machine while other models ran. They are provisional, not idle-device benchmarks.

## Settings and AMI commands

The shared environment used mlx-audio 0.5.6. Workers decode greedily in English.
Granite defaults to 20–30 s chunks and 400 tokens per chunk.
Cohere defaults to energy-based chunks of at most 35 s, batch size 4, and 256 tokens per chunk.
The c15 variant caps energy chunks at 15 s. Historical VAD variants used Silero speech regions capped at 30 or 15 s.
Word times are spread evenly over each decoded chunk, not force-aligned. These tables report WER, not timing accuracy.

From `eval/`, after syncing `envs/granite-cohere/` and preparing AMI:

```sh
uv run python run_granite.py --name granite-4.1-2b --variants sdm --meetings quick -- --orig-pos-attn
uv run python gc_collapse_loops.py --src granite-4.1-2b --dst granite-4.1-2b-noloop --variants sdm
uv run python run_cohere.py --name cohere-2b --variants sdm --meetings quick
uv run python run_cohere.py --name cohere-2b-c15 --variants sdm --meetings quick -- --max-clip 15
uv run python score.py asr --name granite-4.1-2b-noloop --variant sdm --meetings quick
uv run python score.py asr --name cohere-2b --variant sdm --meetings quick
```

`--orig-pos-attn` selects the pre-patch Granite attention used by the table.
Omitting it uses the worker's later patch. Keep those outputs under a different run name.
Loop removal defaults to three retained repeats and can remove genuine repetitions. It is not a supported transcript-cleaning guarantee.
