"""Cross-check: decode given chunks with the reference transformers Cohere Transcribe (no chunking, one call per chunk).

Usage: .venv/bin/python ref_check.py <model_dir> <wav> <start:end> [<start:end> ...] [--device mps] [--dtype bfloat16]
"""

import argparse
import time

import soundfile as sf
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("wav")
    parser.add_argument("spans", nargs="+")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = CohereAsrForConditionalGeneration.from_pretrained(args.model_dir, dtype=dtype).to(args.device).eval()
    audio, sr = sf.read(args.wav, dtype="float32")
    for span in args.spans:
        s, e = (float(x) for x in span.split(":"))
        piece = audio[int(s * sr) : int(e * sr)]
        t = time.perf_counter()
        inputs = processor(piece, sampling_rate=sr, return_tensors="pt", language="en")
        inputs.to(args.device, dtype=dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256)
        kw = {"audio_chunk_index": inputs["audio_chunk_index"]} if "audio_chunk_index" in inputs else {}
        text = processor.decode(out, skip_special_tokens=True, language="en", **kw)
        text = text[0] if isinstance(text, list) else text
        print(f"{span} words={len(text.split())} ({time.perf_counter() - t:.1f}s): {text}", flush=True)


if __name__ == "__main__":
    main()
