# Qwen3-ASR (MLX) + Qwen3-ForcedAligner

**Historical status: dropped for speed (2026-09-24).** The 1.7B run stopped after SDM 8/8 and IHM 4/8 meetings.
The 0.6B run completed SDM 8/8, then stopped without IHM coverage.
The experiment required a large quality gain to justify slower turnaround.
Its IHM gate was approximately WER ≤27.4 at ≥15x real time. These contended timings did not establish idle-machine performance.
See [the evaluation index](../README.md) for protocol, reproduction limits and the pending publication gate.

## Result

- **1.7B** had the lowest WER among the compared AMI SDM runs: 25.62 versus 30.25 for parakeet-v3 at 120/15 s and 27.40 for per-turn parakeet. It lowered cpWER by 2.3–3.5 points with both diarizers. Its contended runtime was 2.4x real time versus 41.6x for per-turn parakeet. These measurements do not establish an uncontended speed ratio.
- **0.6B** was close to per-turn parakeet WER (27.53 versus 27.40), at a contended 6.4x real time.
- The **forced aligner** had median start error 29–38 ms versus parakeet's 80 ms in these runs. However, 8–12% of aligned words had zero length, including some repetition loops.
- Qwen omitted most fillers but retained repetitions. Its measured insertion rate remained low, but short repeated-word loops occurred more often with 0.6B.

## Setup

The current environment omits the optional `[aligner]` dependency extra because exact-release grants for its Japanese/Korean tokenizer packages were not established.
English ASR and forced alignment do not require those tokenizers and remain available.
This changes the install inventory, not the historical measurements below.
See the [dependency evidence and reinstatement conditions](../../docs/licensing/dependencies.md).

- **Path**: `mlx-qwen3-asr==0.4.4` (moona3k; mlx 0.32.2) in its own uv project at `eval/envs/qwen3/`. The worker is `eval/envs/qwen3/qwen3_worker.py` and the runner is `eval/run_qwen3.py`. The package splits audio at low-energy points into chunks of at most 30 s. It decodes each chunk greedily with an adaptive token budget (about 12 tokens/s) and then force-aligns each chunk's text with the native MLX `Qwen/Qwen3-ForcedAligner-0.6B`. The language is forced to English.
- **Weights**: the official `Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3-ASR-0.6B` bf16 checkpoints, loaded as **fp16** (the package default; the package reports that 8-bit gives the same output as fp16). The aligner is also fp16. The cached `aufklarer/Qwen3-ASR-0.6B-MLX-4bit` was not used.
- **Memory fix in the worker**: loading casts bf16 to fp16 and leaves about 4 GB of source buffers in MLX's cache. The worker calls `mx.clear_cache()` after each load. Without this call, the peak footprint was 11.0–11.9 GB on a 75 s clip. With it, the peak is 7.9 GB. A real caller would do the same.
- **Words and sentences**: words retain decoded punctuation. Each chunk's aligned words map back to its punctuated text. Sentences split at `. ? !` and chunk boundaries. The report recorded no unaligned tokens or text mismatches in completed runs. Coverage was SDM 8/8 for both sizes and IHM 4/8 for 1.7B.
- **Alternative implementation, not benchmarked**: `mlx-audio==0.5.6` in `eval/envs/qwen3-mlxaudio/` passed an initial smoke test. Its evaluated defaults used 20-minute chunks and a whole-file `max_tokens=8192` budget, which truncated a 35-minute input. Word alignment needed a separate model and chunking. This report's tables use `mlx-qwen3-asr`, not that path.
- **Contention**: other agents ran parakeet, Nemotron, senko and WeSpeaker on the same GPU at the same time. All runtimes are therefore inflated and provisional. EN2002b on 1.7B (2537 s, including 803 s of alignment) is the clearest outlier. Even on a quiet GPU, the package's own M4 Pro benchmark (RTF 0.077 on a 10 s clip for 1.7B fp16, about 13x real time before alignment) is far below parakeet's speed.

## ASR (WER %, pooled; Whisper normalizer)

| model | variant | meetings | WER | sub | del | ins | runtime s (RTFx) | peak footprint MB |
|---|---|---|---|---|---|---|---|---|
| parakeet-v3 (baseline) | sdm | 8 | 30.25 | 6.50 | 22.12 | 1.63 | ~30–50x | 6576–6918 |
| parakeet-v3-turns (historical comparison) | sdm | 8 | 27.40 | | | | 41.6x | |
| **qwen3-1.7b** | sdm | 8 | **25.62** | 5.72 | 18.27 | 1.63 | 5885 (2.4x) | 7910 |
| qwen3-0.6b | sdm | 8 | 27.53 | 6.22 | 19.79 | 1.52 | 2190 (6.4x) | 5638 |
| parakeet-v3 | ihm | quick 4 | 29.95 | 4.76 | 23.83 | 1.37 | (rerun under contention) | 6048 |
| **qwen3-1.7b** | ihm | quick 4 (stopped) | **24.84** | 4.92 | 17.85 | 2.08 | 1580 (3.5x) | 7894 |

Almost all of the gain comes from fewer deletions: on SDM, Qwen 1.7B has 18.3% deletions against 22.1% for parakeet. The largest per-meeting gains are on the long, clean meetings (ES2004b 17.4, IS1009b 16.1, TS3003b 16.3). EN2002a/b stay at 36–37%, because overlap dominates there.

Aligner share of the time (the worker times it separately; it is included in the totals above):

| model | variant | decode + align s | align s | align share | RTFx without the aligner |
|---|---|---|---|---|---|
| qwen3-1.7b | sdm | 5832 | 1325 | 23% | 3.1x |
| qwen3-0.6b | sdm | 2164 | 410 | 19% | 7.9x |

Model load took 1–7 s. Observed peak footprint was 7.9 GB for 1.7B and 5.2–5.6 GB for 0.6B.
The worker decoded per chunk and cleared the cache between chunks.
The two-hour memory test was not run, so these observations do not establish long-file memory bounds.

## Combo (word merge; cpWER / tcpWER %)

| ASR | diarizer | variant | cpWER | tcpWER |
|---|---|---|---|---|
| parakeet-v3 | senko | sdm (8) | 41.44 | 43.75 |
| qwen3-1.7b | senko | sdm (8) | **39.15** | **40.91** |
| qwen3-0.6b | senko | sdm (8) | 39.88 | 41.99 |
| parakeet-v3 | nemotron3-mlx | sdm (8) | 31.64 | 33.31 |
| qwen3-1.7b | nemotron3-mlx | sdm (8) | **28.16** | **29.61** |
| qwen3-0.6b | nemotron3-mlx | sdm (8) | 30.35 | 31.77 |
| parakeet-v3 | senko | ihm (quick 4) | 34.49 | 36.07 |
| qwen3-1.7b | senko | ihm (quick 4) | 30.35 | 31.66 |
| parakeet-v3 | nemotron3-mlx | ihm (quick 4) | 32.35 | 33.84 |
| qwen3-1.7b | nemotron3-mlx | ihm (quick 4) | 26.85 | 27.95 |

The cpWER gain is larger than the WER gain: 3.5 points against 2.3 on SDM with Nemotron. The gap between tcpWER and cpWER (about 1.5 points) is the same as parakeet's. So the aligner's timestamps are good enough for word-level merge.

## Word timestamps

The AMI manual word times (`ami_public_manual_1.6.2` `words/*.xml`) were matched to the hypothesis words with difflib on lowercased tokens, over exactly matched words only. The script is `eval/analyze_qwen3.py`, and its output is in `runs/results/qwen3_analysis__*.json`.

| run | variant | median dur s | zero-length % | matched % | median \|start err\| s | p90 \|start err\| s | start ≤ 0.2 s % | median \|end err\| s | end ≤ 0.2 s % |
|---|---|---|---|---|---|---|---|---|---|
| parakeet-v3 | sdm | 0.24 | 0.2 | 88.8 | 0.080 | 0.20 | 89.2 | 0.090 | 81.4 |
| qwen3-1.7b | sdm | 0.16 | 12.0 | 90.7 | 0.038 | 0.20 | 89.8 | 0.043 | 79.9 |
| qwen3-0.6b | sdm | 0.16 | 11.3 | 89.4 | 0.038 | 0.19 | 90.3 | 0.042 | 80.2 |
| parakeet-v3 | ihm (quick) | 0.24 | 0.1 | 90.8 | 0.080 | 0.21 | 88.6 | 0.090 | 79.7 |
| qwen3-1.7b | ihm (quick) | 0.16 | 8.2 | 91.5 | 0.029 | 0.12 | 93.5 | 0.034 | 82.8 |

- Median timing error was about half parakeet's in these runs. The parakeet baseline included the historical punctuation extension, now removed from scribe. Tail error (p90) was similar.
- 8–12% of words had zero length (`start == end`). The aligner's LIS repair collapsed some runs onto one timestamp. A zero-length word can inherit the previous speaker in scribe's merge, risking incorrect brief-reply attribution. Giving such words a minimum width was a proposed mitigation, not a validated integration.

## Style (verbatim vs clean)

| run | variant | fillers / 1k words (ref) | immediate repeats / 1k words (ref) |
|---|---|---|---|
| parakeet-v3 | sdm | 35.7 (45.3) | 12.0 (14.7) |
| qwen3-1.7b | sdm | 4.8 (45.3) | 15.1 (14.7) |
| qwen3-0.6b | sdm | 7.3 (45.3) | 14.8 (14.7) |

Qwen writes almost no fillers. The scorer removes fillers anyway, so this has no effect on WER, but the transcripts read "clean". Qwen keeps word repetitions ("the the", "I I") at the reference rate, so it does not lose words to disfluency clean-up. Its deletions are overlap and backchannels, as with parakeet.

## Hallucination

- **Noise and silence probes were not run for Qwen** before the experiment stopped. Planned probes were 300 s each of digital silence, pink noise at −65 dB, and brown noise at −40 dB, plus four AMI SDM edge regions. Probe artifacts may be absent. Parakeet-v3 produced empty text on the three synthetic probes and IS1009b 0–36 s. Other AMI regions contained unannotated setup speech, so they were not true silence. Historical parakeet outputs used `runs/asr/probes/parakeet-v3/`.
- **In-meeting evidence** (words more than 1 s from any reference word): 1.7B has 209 and 0.6B has 216, against 177 for parakeet on SDM. The largest runs are real speech that the reference leaves out, such as ES2004b 15–45 s and IS1009a 0–54 s, which parakeet also transcribes. No chunk hit its token budget (0 truncated chunks).
- **Repetition loops** (an n-gram repeated back-to-back at least 4 times): on SDM, 1.7B has 4 and parakeet has 4, but 0.6B has 12. The reference has 25, mostly real "yeah yeah". One real 1.7B hallucination: EN2002a at about 303.6 s, "Yeah." ×8, all at the same zero-length timestamp at a chunk end, where the reference has one or two. The 0.6B has five 8-repeat loops ("yeah" ×8 ×3 and "sure" ×8 in EN2002a/b). These are short, bounded by the 30 s chunk, and cost few insertions: insertions are 1.5–1.6%, the same as parakeet. But they are the known failure mode of an LLM decoder, and a larger or noisier input could make them worse.

## Files

- `eval/run_qwen3.py`, `eval/envs/qwen3/{pyproject.toml,qwen3_worker.py}`, `eval/analyze_qwen3.py`
- `eval/envs/qwen3-mlxaudio/{pyproject.toml,mlxaudio_worker.py}` (comparison env, not benchmarked)
- Outputs: `eval/runs/asr/qwen3-1.7b/{sdm,ihm}/`, `eval/runs/asr/qwen3-0.6b/sdm/` (`<M>.raw.json` keeps the chunks and the aligned words), `eval/runs/results/`

## AMI commands

From `eval/`, after syncing `envs/qwen3/` and preparing AMI:

```sh
uv run python run_qwen3.py --name qwen3-1.7b --model Qwen/Qwen3-ASR-1.7B --dtype float16 --variants sdm --meetings all
uv run python run_qwen3.py --name qwen3-0.6b --model Qwen/Qwen3-ASR-0.6B --dtype float16 --variants sdm --meetings all
uv run python run_qwen3.py --name qwen3-1.7b --model Qwen/Qwen3-ASR-1.7B --dtype float16 --variants ihm --meetings quick
uv run python score.py asr --name qwen3-1.7b --variant sdm --meetings all
uv run python score.py combo --asr qwen3-1.7b --diar nemotron3-mlx --merge word --variant sdm --meetings all
uv run python analyze_qwen3.py --names parakeet-v3,qwen3-1.7b,qwen3-0.6b --variants sdm --meetings all
```

The combo and analysis commands need their named baseline outputs.
No automated command in this report regenerates the synthetic probe inputs.
Do not treat absent Qwen silence tests as evidence that it avoids hallucinations.
