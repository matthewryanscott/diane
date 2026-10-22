"""ASR candidate: Cohere Transcribe 03-2026 (2B) on MLX via mlx-audio (text only, approximate word times).

Weights: littoralai/cohere-transcribe-mlx-fp16, an mlx-audio conversion of the gated
CohereLabs/cohere-transcribe-03-2026 (Apache-2.0; our token has no access to the
gated repo). mlx-audio chunks internally: <=35 s, cut at the quietest 100 ms window in
the last 5 s of each chunk (the upstream feature extractor's scheme).
Decoding is greedy, English, punctuated.
Word times are spread evenly over each chunk ("word_times": "approximate").

Usage: uv run python run_cohere.py [--name cohere-2b] [--variants sdm,ihm] [--meetings all]
       [--audio FILE] [--force] [-- extra gc_worker.py args, e.g. --batch-size 4]
"""

from run_granite import run

if __name__ == "__main__":
    run("cohere", "cohere-2b", "littoralai/cohere-transcribe-mlx-fp16", __doc__)
