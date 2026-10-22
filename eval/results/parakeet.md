# parakeet: chunking, per-turn decoding, and v2 vs v3

This historical experiment compares parakeet chunking, per-turn decoding, and `parakeet-tdt-0.6b-v2` against `v3`.
AMI results cover eight public meetings (3.9 h per variant).
See [the evaluation index](../README.md) for protocol, reproduction limits and the pending publication gate.

## Historical conclusion

On these AMI meetings, **v3 with senko-derived per-turn windows** reduced errors against the 120/15 s baseline.
The runner used `--diar senko --cover speech --max-window 25`.
Nemotron then supplied speaker attribution, giving cpWER 28.47 (SDM) and 23.49 (IHM-mix), against 31.64 and 27.39.
Plain WER fell from 30.25 to 27.40 (SDM) and from 26.20 to 22.77 (IHM).
Solo ASR speed was 67x real time against 61x, with peak memory 3.0 GB against 6.0 GB.

Per-turn ASR must wait for diarization. Current scribe instead runs ASR and diarization in parallel with **v3 at 30/5 s**.
That fixed-window experiment reached cpWER 28.87 / 23.93 with Nemotron, about 0.4 points above the per-turn result.
These comparisons support the measured settings, not a universal ranking. V2 won at 120/15 s but lost to the best v3 settings.

## Method

- **Fixed chunks (CLI)**: `run_parakeet.py` with parakeet-mlx 0.5.2 CLI arguments. The settings are the default 120/15 s (chunk/overlap), 120/30, 60/15, 30/5, and `--chunk-duration 0 --local-attention` (no chunking). Full attention without chunking was not run: the attention matrix for a 39-min meeting (~29k encoder frames) would need tens of GB. The local-attention run already peaks at 19.9 GB.
- **Per-turn windows**: `run_parakeet_turns.py` builds windows from a diarizer RTTM, and `parakeet_turns_worker.py` decodes them all in one process with the parakeet-mlx Python API (greedy, bf16, one `generate()` per window). There are two cover modes:
  - `speech` (the "turns" runs) takes the union of diarized speech, merges gaps of 0.5 s or less, pads each region by 0.25 s, and splits it to windows of at most 25 s. Splits go at the middle of a non-speech gap first, then at a speaker change, then as a hard cut. Speech that the diarizer misses is never decoded.
  - `all` (the "cuts" runs) splits the whole file the same way, so nothing is skipped and the diarizer only picks the cut points.
  - Windows come from senko (`runs/diar/senko/`) unless the run name ends in `-nem`, which means `nemotron3-mlx`. senko gives 50–150 windows per meeting and Nemotron gives 270–400, because its turns are shorter.
- **Scoring**: `score.py asr` (WER) and `score.py combo --merge word` (cpWER) with both diarizers. The dropped-region analysis is `analyze_gaps.py`. It finds hypothesis gaps of 5 s or more that contain at least 10 normalized reference words, using word times from the AMI `words.xml`.
- **Runtime**: the full-run runtimes shared the machine with other agents' model runs, so they are provisional. The solo re-timing ran two SDM meetings (IS1009a and ES2004b, 3184 s of audio) one after another on an otherwise idle machine. Peak memory (footprint) is reliable in both cases.

## Results

### WER (pooled, 8 meetings) and peak memory

| setting | SDM WER (sub/del/ins) | IHM WER (sub/del/ins) | peak MB | full-run RTFx SDM / IHM (provisional) |
|---|---|---|---|---|
| v3 default 120/15 | 30.25 (6.50/22.12/1.63) | 26.20 (4.65/19.93/1.62) | 6679 | 30 / 15 |
| v3 120/30 | 29.17 (6.66/20.81/1.71) | 25.22 (4.88/18.65/1.69) | 6981 | 11 / 13 |
| v3 60/15 | 28.23 (6.47/19.99/1.77) | 24.12 (4.71/17.81/1.61) | 5088 | 23 / 16 |
| v3 30/5 | 27.70 (6.31/19.53/1.87) | 22.84 (4.70/16.33/1.81) | 3964 | 20 / 18 |
| v3 no chunking, local attention | 28.47 (6.51/20.34/1.62) | 24.11 (4.65/17.84/1.62) | 19930 | 20 / 19 |
| **v3 turns (senko, speech)** | **27.40** (6.34/19.31/1.74) | **22.77** (4.52/16.40/1.85) | 3416 | 42 / 29 |
| v3 cuts (senko, all) | 27.31 (6.28/19.22/1.82) | 23.04 (4.59/16.61/1.84) | 3648 | 19 / 21 |
| v3 turns (Nemotron, speech) | 27.68 (7.97/17.41/2.30) | 23.45 (5.31/16.07/2.06) | 3169 | 45 / 45 |
| v2 default 120/15 | 28.13 (6.11/20.40/1.62) | 23.99 (4.47/18.10/1.42) | 6546 | 15 / 56 |
| v2 30/5 | 29.07 (6.13/21.28/1.67) | 24.57 (4.36/18.77/1.44) | 4278 | 71 / 71 |
| v2 turns (senko) | 29.31 (6.33/21.28/1.69) | 24.94 (4.26/19.14/1.54) | 3758 | 70 / 76 |
| v2 turns (Nemotron) | 29.87 (8.34/19.44/2.09) | 25.24 (5.41/18.01/1.82) | 3058 | 51 / 52 |

For v3, shorter windows cut deletions by about 3 points on both variants, and substitutions stay flat. Nemotron-cut windows are very short (median 2.7 s, and 72% are under 5 s, against a median of 16.2 s for senko), so they trade deletions for substitutions and insertions (+1.6 sub on SDM). Senko-cut windows are the better cut source even when Nemotron does the speaker attribution.

### cpWER (word merge), by diarizer

| setting | SDM + senko | SDM + Nemotron | IHM + senko | IHM + Nemotron |
|---|---|---|---|---|
| v3 default 120/15 | 41.44 | 31.64 | 30.48 | 27.39 |
| v3 60/15 | 40.28 | 30.42 | 29.10 | 26.35 |
| v3 30/5 | 39.60 | 28.87 | 27.59 | 23.93 |
| v3 no chunking, local attention | 39.43 | 29.66 | 27.93 | 24.84 |
| **v3 turns (senko)** | **39.03** | 28.47 | **26.79** | **23.49** |
| v3 cuts (senko) | 39.20 | **28.41** | 27.25 | 23.89 |
| v3 turns (Nemotron) | 39.80 | 28.73 | 27.60 | 24.11 |
| v2 default | 39.48 | 29.53 | 28.09 | 25.16 |
| v2 30/5 | 40.39 | 30.25 | 28.59 | 25.57 |
| v2 turns (senko) | 39.71 | 30.37 | 28.37 | 25.69 |
| v2 turns (Nemotron) | 41.30 | 30.94 | 29.01 | 25.79 |

### Solo speed (idle machine, IS1009a + ES2004b SDM, 3184 s of audio)

| setting | runtime s | RTFx | peak MB | WER on these 2 |
|---|---|---|---|---|
| v3 default 120/15 | 52.5 | 61 | 5987 | 23.09 |
| v3 60/15 | 58.0 | 55 | 4939 | 20.23 |
| v3 30/5 | 52.6 | 61 | 4147 | 19.84 |
| v3 turns (senko) | 47.3 | 67 | 2978 | 20.96 |
| v2 default | 44.6 | 71 | 5950 | 20.71 |
| v2 30/5 | 44.4 | 72 | 4039 | 20.93 |
| v2 turns (senko) | 41.2 | 77 | 2731 | 22.55 |

Model load is included in every runtime. Per-turn runtimes exclude the diarizer, which must finish first.
The solo settings ran at 55–77x real time on this two-meeting subset. V2 was about 15% faster than v3.
Per-turn decoding was about 10% faster than chunking, partly because it skipped non-speech.
Nemotron-cut runs were not solo-timed. Their contended full-run timings cannot establish a reliable speed ranking.

## Dropped regions

A dropped region is a hypothesis gap of 5 s or more that contains at least 10 reference words. The last column counts the regions that touch a chunk-overlap zone. By chance, about a quarter of the regions would.

| setting | SDM regions | SDM ref words (% of all) | IHM regions | IHM ref words (%) | touching an overlap zone (SDM / IHM) |
|---|---|---|---|---|---|
| v3 default 120/15 | 65 | 1759 (4.80) | 52 | 1515 (4.14) | 14 / 10 |
| v3 120/30 | 52 | 1293 (3.53) | 37 | 1067 (2.91) | 19 / 9 |
| v3 60/15 | 52 | 1248 (3.41) | 35 | 924 (2.52) | 39 / 16 |
| v3 30/5 | 51 | 1098 (3.00) | 19 | 480 (1.31) | 35 / 10 |
| v3 no chunking, local attention | 47 | 850 (2.32) | 29 | 728 (1.99) | - |
| v3 turns (senko) | 45 | 1114 (3.04) | 24 | 613 (1.67) | - |
| v3 cuts (senko) | 42 | 1005 (2.75) | 31 | 724 (1.98) | - |
| v3 turns (Nemotron) | 17 | 408 (1.11) | 18 | 438 (1.20) | - |
| v2 default | 49 | 1045 (2.85) | 23 | 535 (1.46) | 10 / 5 |
| v2 30/5 | 66 | 1566 (4.28) | 39 | 843 (2.30) | 38 / 20 |
| v2 turns (senko) | 65 | 1641 (4.48) | 39 | 1279 (3.49) | - |
| v2 turns (Nemotron) | 34 | 973 (2.66) | 29 | 1043 (2.85) | - |

The findings are as follows:

- **v3 at the default silently drops whole passages of clear, single-speaker speech.** The worst SDM cases are:
  - TS3003a 361–423 s, 123 reference words.
  - ES2004b 1015–1051 s, 91 words.
  - TS3003a 0–66 s, 67 words.
  - TS3003b 405–429 s, 49 words.

  Other tested settings recovered these passages almost completely.
- **The model causes the drops, not the chunk merge.** Decoding the single 120 s window [315, 435] s of TS3003a on its own gives no tokens from 360.6 s to 422.6 s. The same audio in the window [330, 450] s, shifted by 15 s, is transcribed in full. Likewise, the window [945, 1065] s of ES2004b emits nothing after 1014.8 s, and the window [0, 120] s of TS3003a emits nothing for its first 60 s. Once the decoder stops emitting inside a long window, it can stay silent up to the window's end. The overlap merge cannot recover this, because only 15 s of each window is decoded twice.
- **Drops touch the overlap zones at about chance rate** (14 of 65 for the default, against ~25% expected), which again points at the decoder rather than the merge. The 60/15 and 30/5 runs touch overlap zones more often only because those zones cover more of the timeline.
- **Shorter windows make drops rarer and cap how much each one can lose.** No setting removes them completely, and the regions each setting drops are different (for example, 60/15 drops ES2004b 976–989 s and 30/5 drops ES2004b 1540–1548 s). Long dropped regions explain about 1.5–2 of the ~3-point cut in deletions. The rest comes from shorter dropouts under the 5 s / 10-word threshold.
- **Most deletions are not avoidable with chunking.** Even the best setting keeps 16–19% deletions, and dropped regions hold only 1–3% of reference words. The rest is overlapped speech and backchannels that single-stream ASR does not transcribe.
- **v2 behaves the other way round.** It drops less than v3 at 120 s (2.85% against 4.80% of words on SDM) but more with short windows (4.3–4.5%). That is why v2 gets worse with shorter windows.

## Trade-offs of per-turn decoding

- ASR depends on diarization. It can no longer run in parallel with it, and diarizer misses become ASR deletions in `speech` mode. `cuts` mode (`--cover all`) avoids the second problem at about the same WER (SDM 27.31 against 27.40, IHM 23.04 against 22.77) and costs ~10% more decode time.
- Very short windows hurt accuracy. Nemotron's turns (270–400 windows per meeting) raise substitutions by 1–1.6 points compared with senko's turns (50–150). If Nemotron is the only diarizer, merge its turns into longer windows (for example, only split at gaps or when a window reaches 25 s) before decoding. Alternatively, use `--cover all`, where the minimum window length is a third of the maximum.
- Sentence boundaries fall at window boundaries, which suits scribe's word-level merge.

## AMI commands

From `eval/`, after preparing AMI and the candidate environments:

```sh
uv run python run_senko.py --meetings all
uv run python run_nemotron_mlx.py --meetings all
uv run python run_parakeet.py --name parakeet-v3 --meetings all
uv run python run_parakeet.py --name parakeet-v3-c30 --meetings all -- --chunk-duration 30 --overlap-duration 5
uv run python run_parakeet_turns.py --name parakeet-v3-turns --diar senko --cover speech --max-window 25 --meetings all
uv run python score.py asr --name parakeet-v3-turns --meetings all
uv run python score.py combo --asr parakeet-v3-turns --diar nemotron3-mlx --merge word --meetings all
uv run python analyze_gaps.py --names parakeet-v3,parakeet-v3-c30,parakeet-v3-turns --variant both
```

Use a distinct run name for each setting. Runners skip existing outputs unless given `--force`.
The historical package was parakeet-mlx 0.5.2. Current runners resolve scribe's version range and current token conversion.

## Files

- `eval/run_parakeet_turns.py` and `eval/parakeet_turns_worker.py` are the per-turn runner and worker.
- `eval/analyze_gaps.py` is the dropped-region analysis. Its summary goes to `runs/results/gaps__{sdm,ihm}.json`.
- Historical outputs used `runs/asr/parakeet-v3{,-c60,-c30,-c120o30,-c0local,-turns,-cuts,-turns-nem}` and `runs/asr/parakeet-v2{,-c30,-turns,-turns-nem}`. Solo timings used `runs/asr/timing-*` on IS1009a and ES2004b SDM. Raw token and timing artifacts are ignored and may be absent. The report does not provide a dedicated command for the isolated-window decoder checks.
