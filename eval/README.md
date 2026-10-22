# Meeting speech-model evaluation (AMI)

This harness compares local ASR and diarization candidates for recurring 2–6-person meeting transcription.
It scores AMI meetings with WER, DER, JER, cpWER and tcpWER after scribe's `merge()`.
The tables record historical experiments, not accuracy guarantees for current models or private meetings.
The supported product transcribes and processes meetings. These experiments do not establish extraction or summary support.

## Layout

```
eval/
  pyproject.toml, uv.lock   own uv project (Python 3.13); lock freely
  common.py                 paths, meeting lists, RTTM/JSON helpers; puts ../scribe on sys.path
  prepare_ami.py            downloads audio and builds references
  measure.py                wraps a command in /usr/bin/time -l (runtime + peak footprint)
  run_parakeet.py           ASR baseline "parakeet-v3"
  run_senko.py              diarizer baseline "senko" (drives senko_worker.py in the main venv)
  score.py                  asr | diar | combo scorers
  data/ami/                 (ignored) audio/{sdm,ihm}/<M>.wav, ref/{seglst,rttm,uem,rttm_from_words}/, meetings.json, raw/
  runs/                     (ignored) asr/<name>/<variant>/, diar/<name>/<variant>/, combo/, results/
```

## Data

The data is the 8 AMI full-corpus-ASR test meetings. The "quick" subset is the four "a" meetings. `sdm` is Array1-01 (a single distant mic) and `ihm` is Mix-Headset (a mix of the headset mics, so overlaps are still mixed). All audio is 16 kHz mono wav.

| meeting | duration s | speakers | ref segments | ref words (raw) | quick |
|---|---|---|---|---|---|
| ES2004a | 1049.4 | 4 | 260 | 2614 | yes |
| ES2004b | 2345.5 | 4 | 467 | 6763 | |
| IS1009a | 838.8 | 4 | 195 | 1986 | yes |
| IS1009b | 2052.3 | 4 | 389 | 6091 | |
| TS3003a | 1505.6 | 4 | 242 | 2518 | yes |
| TS3003b | 2210.3 | 4 | 404 | 4836 | |
| EN2002a | 2142.7 | 4 | 746 | 7516 | yes |
| EN2002b | 1786.8 | 4 | 488 | 6002 | |

The total is 13931 s (3.9 h) per variant.

References:

- **SegLST** (`ref/seglst/<M>.json`) comes from the official `ami_public_manual_1.6.2` annotations. There is one segment per AMI manual segment (`segments/*.segments.xml`). A segment holds only the real `<w>` words: punctuation, vocal sounds, disfluency markers and gaps are dropped, and truncated words are kept. Segment times are the first word start and the last word end. The speaker is the AMI global speaker ID, the same as in the RTTM.
- **RTTM for DER** (`ref/rttm/`) is the pyannote AMI-diarization-setup `only_words` test RTTM, scored inside its UEM (`ref/uem/`). As a cross-check, `ref/rttm_from_words/` is derived from the SegLST. It matches the pyannote RTTM to a pooled DER of 0.02% (0.00% on 7 of the 8 meetings).

## Adapter contract (to add a candidate)

- **ASR** writes `runs/asr/<name>/<variant>/<meeting>.json` as `{"words":[{"text","start","end"}...], "sentences":[{"text","start","end","words":[...]}] (optional), "meta":{"model","settings","runtime_s","peak_mem_mb","audio_s"}}`. `<variant>` is `sdm` or `ihm`. Without `sentences`, combo `--merge sentence` is not available, and `--merge word` treats all words as one segment.
- **Diarizer** writes `runs/diar/<name>/<variant>/<meeting>.rttm` and `<meeting>.meta.json` (the same meta fields). It can also save `.npy` embeddings or centroids next to them. senko saves `<M>.centroids.npy` and `<M>.speakers.json`.
- **Measurement** uses `measure.measure(cmd, log=...)`, or the CLI `python measure.py --out meta.json -- <cmd>`. It returns `runtime_s`, `peak_mem_mb` ("peak memory footprint", which includes Metal/unified memory) and `max_rss_mb`. Wrap the real model process. Do not wrap `uv run` or `uvx`: they spawn a child, and the footprint that comes back is uv's own ~10 MB. Resolve the interpreter or executable first (see `run_parakeet.py` for uvx and `run_senko.py` for the main venv). Run one process per meeting so that load time is included, and run candidates one after another, never at the same time.
- To add a candidate, copy `run_parakeet.py` or `run_senko.py`, change the command and the output conversion, and run it with `--meetings quick` first.

## Commands

Run these commands from the repository root. Preparation downloads the AMI inputs. Run candidates sequentially.
The first sync prepares the root `.venv` for senko. The second sync prepares the evaluation environment.

```sh
uv sync
cd eval
uv sync
uv run python prepare_ami.py
uv run python run_parakeet.py --name parakeet-v3 --variants sdm,ihm --meetings all
uv run python run_senko.py --name senko --variants sdm,ihm --meetings all
uv run python score.py asr --name parakeet-v3 --variant both --meetings all
uv run python score.py diar --name senko --variant both --meetings all
uv run python score.py combo --asr parakeet-v3 --diar senko --merge both --variant both --meetings all
uv run python score.py -v combo --asr parakeet-v3 --diar senko --merge word --variant sdm --meetings quick
```

Runners accept `--meetings quick|all` or a comma-separated AMI meeting list.
Scorers use singular `--variant sdm|ihm|both`. Runners use plural `--variants sdm,ihm`.
Pass parakeet model options after `--`, for example:

```sh
uv run python run_parakeet.py --name parakeet-v3-c30 -- --chunk-duration 30 --overlap-duration 5
```

The baseline used parakeet-mlx 0.5.2. The runner resolves scribe's version range, so record the resolved version before comparing new runs.
Fresh runs use the working-tree merge and token conversion. They need not reproduce historical scores after code or dependency changes.

Each scorer prints markdown tables and writes `runs/results/<kind>__<names>__<variant>__<subset>.json`, with per-meeting and pooled values.

Metric definitions:

- **WER** uses the Whisper `EnglishTextNormalizer` on both sides, which also removes fillers (um, uh, hmm, mm). Reference utterances are joined in start-time order and compared with the whole hypothesis through meeteval SISO. Pooled WER is the total errors divided by the total reference words.
- **DER** comes from pyannote.metrics inside the UEM, with overlap scored. "c0" means no collar. "c.25" means ±0.25 s around each reference boundary (pyannote `collar=0.5`, the same as md-eval `-c 0.25`). JER is computed with collar 0. The speaker-count error is hyp − ref.
- **Combo** builds `scribe.TranscriptionSegment`s from the ASR `sentences`: with their words for `word` mode, and with `words=()` for `sentence` mode (the old merge). It runs them through `scribe.merge` with the diarizer RTTM, normalizes each merged segment, and writes the SegLST hypothesis to `runs/combo/`. It then scores meeteval cpWER, tcpWER (collar 5 s, default pseudo-word timing) and greedy ORC-WER. An `UNKNOWN` speaker from merge counts as one more hypothesis speaker.

## Historical baseline results (all 8 meetings, pooled)

The ASR baseline was `parakeet-v3`: the parakeet-mlx 0.5.2 default model `mlx-community/parakeet-tdt-0.6b-v3`.
It used 120 s chunks, 15 s overlap, greedy decoding and bf16.
The diarizer baseline was `senko` at the scribe-pinned commit, with `device="auto"`.
Current scribe uses 30/5 s parakeet windows and Nemotron by default.

| metric | SDM | IHM-mix |
|---|---|---|
| ASR WER (sub/del/ins) | 30.25 (6.50/22.12/1.63) | 26.20 (4.65/19.93/1.62) |
| DER collar 0 (miss/FA/conf) | 33.02 (16.26/7.71/9.04) | 28.00 (15.75/7.70/4.55) |
| DER collar 0.25 | 24.20 | 19.29 |
| JER | 41.71 | 35.61 |
| speaker count: exact, mean abs error | 5/8, 0.62 | 3/8, 0.75 |
| cpWER word merge / sentence merge | **41.44** / 44.18 | **30.48** / 33.52 |
| tcpWER word / sentence | **43.75** / 46.74 | **31.95** / 35.95 |
| greedy ORC-WER word / sentence | 30.76 / 30.87 | 26.64 / 26.80 |

Word-level merge lowered cpWER by 2.7 to 3.0 points on both variants in this run.
The largest gains were on TS3003b (IHM: 23.4 vs 33.7) and TS3003a.
ORC-WER changed little, which is consistent with a speaker-attribution gain rather than a word-recognition gain.

The per-meeting speaker counts from senko are as follows:

- SDM: IS1009a 5, TS3003a 2, EN2002a 2. The rest are 4.
- IHM: ES2004b 5, IS1009a 5, TS3003a 2, EN2002a 5, EN2002b 3.

EN2002a on SDM is the worst case: only 2 speakers found, DER 59%, cpWER 78%.

Runtime and peak memory (each process runs once per meeting, and model load is included):

| candidate | variant | total runtime s (RTFx) | per-meeting runtime s | peak footprint MB (max) |
|---|---|---|---|---|
| parakeet-v3 | sdm | 276.8 (50x) | 14–46 | 6918 |
| parakeet-v3 | ihm | 250.5 (56x) | 16–45 | 6597 |
| senko | sdm | 102.6 (136x) | 9–16 | 1303 |
| senko | ihm | 131.7 (106x) | 12–20 | 1256 |

senko's own `total_time` is only 31 s (SDM) and 39 s (IHM) over all 8 meetings, at most 8 s per meeting. The rest of the process time is imports and model load, which are recorded in `<M>.meta.json` under `senko.timing_stats`. The peak footprint of parakeet (5.2–6.9 GB) is mostly full-attention activations on 120 s chunks, while its max RSS is only ~0.9 GB. This shows why footprint, not RSS, is the right measure on Apple silicon.

## Caveats

- Deletions dominate the ASR errors, even on IHM (about 20% deletions against about 1.6% insertions). AMI has many backchannels and overlapping speakers. A single-stream ASR can miss quieter speakers. Treat these results as specific to these meetings, references and settings.
- WER joins reference utterances in start-time order, so word order inside overlaps can differ from the hypothesis. All candidates use this protocol, but their outputs can be affected differently.
- The historical `_words_from_tokens` extended a word's end time through trailing punctuation, sometimes by about 0.5 s. Current scribe no longer does this. Historical merge and timing scores can therefore differ from fresh runs.
- The combo hypothesis uses merged segments, not single words. tcpWER uses meeteval's default pseudo-word timing (`character_based_points` for the hypothesis and `character_based` for the reference). meeteval reports small hypothesis self-overlap (~0.1%) and ordering warnings for these segments. They are hidden unless `-v` is given. As a sanity check, the reference scored against itself gives 0 cpWER and 0 tcpWER.
- ORC-WER is the greedy variant (`greedy_orcwer`), because exact ORC-WER is exponential in the number of hypothesis speakers.
- Runtimes come from one sequential pass on an M5 with 32 GB of RAM and are not averaged over repeats. Allow a few percent of noise.
- `import scribe` resolves to the working-tree `scribe/scribe.py` (through `common.py`), including the senko worker that runs in the main `.venv`.

## Experiment reports and disposition

The seven Markdown files here retain public AMI methods, historical results and limitations. Private-input helpers and their reports are excluded from this candidate, not anonymized into public evidence.

| file | disposition |
|---|---|
| `README.md` | Retain the AMI protocol, baseline tables, commands and reproducibility caveats. |
| `results/parakeet.md` | Retain AMI chunking, decoding and dropped-region evidence; exclude private long-file results. |
| `results/nemotron3.md` | Retain AMI port comparisons; exclude private-proxy, biometric and long-file results. |
| `results/qwen3.md` | Retain AMI quality, timing, alignment and repetition evidence. State incomplete coverage and contention. |
| `results/pyannote.md` | Retain AMI comparisons and incomplete-run limits. Remove account identity and obsolete workflow references. |
| `results/granite_cohere.md` | Retain quick-subset evidence and patch uncertainty. Remove private attribution and obsolete workflow references. |
| `results/diarizen.md` | Retain partial AMI comparisons, device tests, licensing and runtime limits. Correct subset claims. |

Candidate reports are in [results](results/). This candidate excludes `diane_proxy/`, `voiceprint/`, `nemotron_diane.py` and `run_parakeet_diane.py`, along with private-input modes in otherwise public AMI runners.
Voiceprint matching remains a feature under evaluation. It is not a universal identity or accuracy guarantee.
`merge_refine.py` holds the merge-smoothing grid. `run_scribe_e2e.py` scores scribe end to end.

## Reproduction and publication limits

AMI experiments have downloadable inputs and reference-building code in `prepare_ami.py`.
Reproduction still depends on corpus access, licenses, model revisions, dependency versions and the working-tree implementation.
Private recordings, labels, identity-linked derived results and their reproduction helpers are not included.
Do not substitute private recordings for AMI inputs in the examples.

The FluidAudio experiment needs an uncommitted source checkout:

- `envs/fluidaudio/`: FluidAudio at `3fc97306`. Build with `swift build -c release --product fluidaudiocli` in that directory. The historical SwiftPM artifact workaround is in [nemotron3.md](results/nemotron3.md).

Other candidate environments keep `pyproject.toml` and `uv.lock`. Run `uv sync` inside the selected environment.
`prepare_ami.py` downloads and converts AMI audio. Ignored audio, model caches and run artifacts may be absent.
The forced-alignment comparison needs separate references described in [nemotron3.md](results/nemotron3.md).

The optional DiariZen runner, worker and environment are excluded from this source candidate.
The Cohere runner no longer exposes the optional MLX Silero VAD path.
Their [historical results](results/) remain evidence of past experiments, not supported reproduction commands.
See the [licensing exclusions and primary evidence](../docs/licensing/models-and-data.md) before using archived inputs.

Retaining public AMI results does not authorize publication of private source archives or imply participant consent.
A separate release gate must review corpus and model licenses, scripts, and generated artifacts.
That review includes transcripts, labels, embeddings, logs and cached outputs, not only these Markdown files.
