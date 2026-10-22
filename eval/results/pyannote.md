# pyannote speaker-diarization-community-1

This historical report summarizes runs from 2026-09-24 in `eval/runs/diar/pyannote-c1*`.
The runner is `eval/run_pyannote.py`, with `eval/pyannote_worker.py` and `eval/envs/pyannote/`.
It used pyannote.audio 4.x, torch 2.14 and MPS.
The model requires gated access. Prior access does not authorize another account or publication.
See [the evaluation index](../README.md) for protocol, reproduction limits and the pending publication gate.

## Result

The experiment did not adopt pyannote.
It had lower DER against the harness references, but higher cpWER and fewer exact speaker counts than Nemotron on these meetings.
Measured speed was about 34x slower under contention. That ratio is not an idle-machine benchmark.

AMI, 8 meetings, pooled:

| | Nemotron 3 (mlx) SDM | pyannote SDM | pyannote exclusive SDM | Nemotron 3 (mlx) IHM | pyannote IHM | pyannote exclusive IHM |
|---|---|---|---|---|---|---|
| DER collar 0 / ±0.25 s | 27.11 / 24.62 | 20.22 / 12.99 | 24.55 / 16.32 | 25.80 / 23.77 | 17.06 / 10.42 | 21.93 / 13.99 |
| cpWER (parakeet-v3, word merge) | 31.64 | 35.27 | 34.46 | 27.39 | 30.59 | 29.43 |
| speaker count exact | 7/8 | 5/8 | 5/8 | 7/8 | 5/8 | 5/8 |
| speed (x real time) | 250 | 7.3 | (same run) | 236 | 6.5 | (same run) |
| peak memory | 1.7 GB | 4.4 GB | | 1.6 GB | 5.1 GB | |

## Notes

- Why DER and cpWER disagree: pyannote (like DiariZen) fills pauses inside a turn, which matches the harness's reference segments; the transcript's speaker labels still come out worse. Nemotron's remaining DER is mostly missed speech in pauses, which the word merge ignores.
- The `exclusive_speaker_diarization` output improves cpWER by ~1 point over the regular output, but not enough.
- Timing: the speaker-embedding step dominates (e.g. ES2004a: segmentation 10 s, embeddings 117 s). Runs were measured while other models used the GPU; pyannote's published speed on an idle M4 Pro is ~24x, still ~10x slower than Nemotron.
- Incomplete AMI checks: speaker-count hints (`pyannote-c1-oracle`, `pyannote-c1-2to6`) provide no completed comparison.
- The pipeline returns 256-dimensional speaker embeddings, saved as `.embeddings.npy`. These results do not establish their cross-recording identity accuracy.

## AMI commands

From `eval/`, after syncing `envs/pyannote/`, obtaining authorized model access, and preparing AMI:

```sh
uv run python run_pyannote.py --name pyannote-c1 --device mps --variants sdm,ihm --meetings all
uv run python score.py diar --name pyannote-c1 --meetings all
uv run python score.py combo --asr parakeet-v3 --diar pyannote-c1 --merge word --meetings all
```

The combo command needs the historical parakeet baseline outputs.
The runner also writes exclusive diarization under `pyannote-c1-excl`. Score that name to compare exclusive output.
