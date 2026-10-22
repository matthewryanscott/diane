"""Transcribe a list of time windows of one wav with parakeet-mlx, loading the model once.

Runs under the parakeet-mlx uvx environment (stdlib + parakeet_mlx only). Each window
is decoded on its own (no chunk merging); token times are shifted to file time, and
each window's sentences are kept as they are, so sentence boundaries fall at window
boundaries. The output has the parakeet-mlx CLI JSON shape (sentences[].tokens[]).

Usage: <parakeet-env>/bin/python parakeet_turns_worker.py <wav> <windows.json> <out.json> [--model M]
<windows.json> is [[start_s, end_s], ...].
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from parakeet_mlx import AlignedResult, AlignedSentence, DecodingConfig, from_pretrained
from parakeet_mlx.audio import get_logmel, load_audio
from parakeet_mlx.cli import to_json

MIN_WINDOW_S = 0.1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    parser.add_argument("windows", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v3")
    args = parser.parse_args()

    t0 = time.perf_counter()
    model = from_pretrained(args.model, dtype=mx.bfloat16)
    cfg = model.preprocessor_config
    audio = load_audio(args.wav, cfg.sample_rate, mx.bfloat16)
    t_load = time.perf_counter() - t0

    windows = json.loads(args.windows.read_text())
    sentences: list[AlignedSentence] = []
    for start, end in windows:
        if end - start < MIN_WINDOW_S:
            continue
        a, b = int(start * cfg.sample_rate), int(end * cfg.sample_rate)
        mel = get_logmel(audio[a:b], cfg)
        result = model.generate(mel, decoding_config=DecodingConfig())[0]
        for sentence in result.sentences:
            for token in sentence.tokens:
                token.start += start
                token.end = token.start + token.duration
            sentences.append(AlignedSentence(text=sentence.text, tokens=sentence.tokens))
        mx.clear_cache()  # windows differ in shape; don't let the buffer cache inflate the footprint
    t_decode = time.perf_counter() - t0 - t_load

    text = "".join(s.text for s in sentences)
    data = json.loads(to_json(AlignedResult(text=text, sentences=sentences)))
    data["timing"] = {"load_s": round(t_load, 2), "decode_s": round(t_decode, 2), "windows": len(windows)}
    args.out.write_text(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    main()
