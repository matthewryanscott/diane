"""Silero VAD speech segments for mono 16 kHz wavs (per-channel ground truth for run_stereo.py).

Runs in a throwaway env: uv run --no-project --python 3.13 --with silero-vad --with numpy python stereo_vad_worker.py OUT_DIR WAV...
Writes OUT_DIR/<wav stem>.json as [[start_s, end_s], ...].
"""

import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

torch.set_num_threads(4)
out_dir = Path(sys.argv[1])
model = load_silero_vad()
for wav in map(Path, sys.argv[2:]):
    out = out_dir / f"{wav.stem}.json"
    if out.exists():
        continue
    with wave.open(str(wav), "rb") as wf:
        audio = torch.from_numpy(np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768)
    stamps = get_speech_timestamps(audio, model, sampling_rate=16000, return_seconds=True)
    out.write_text(json.dumps([[s["start"], s["end"]] for s in stamps]))
    print(wav.stem, len(stamps), flush=True)
