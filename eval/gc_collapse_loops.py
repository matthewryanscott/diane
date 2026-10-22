"""Post-process a Granite/Cohere run: cut greedy repetition loops out of each chunk, write it as a new run name.

A loop is an n-gram (n = 1..12 words) repeated back to back more than --max-repeats times; the
chunk text is kept up to the end of the last allowed repeat and the rest of the loop is dropped
(text after the loop, if any, is kept). No model is rerun: this reads <src>/<variant>/<M>.raw.json.

Usage: uv run python gc_collapse_loops.py --src granite-4.1-2b --dst granite-4.1-2b-noloop [--variants sdm,ihm]
"""

import argparse

from common import asr_run_dir, read_json, write_json
from run_granite import to_contract


def collapse(tokens: list[str], max_repeats: int) -> tuple[list[str], int]:
    """Return tokens with loops cut, and the number of dropped tokens."""
    out: list[str] = []
    i = dropped = 0
    norm = [t.lower().strip(".,?!") for t in tokens]
    while i < len(tokens):
        cut = 0
        for n in range(1, 13):
            gram = norm[i : i + n]
            if len(gram) < n:
                break
            reps = 1
            while norm[i + reps * n : i + (reps + 1) * n] == gram:
                reps += 1
            if reps > max_repeats:
                cut = max(cut, reps * n)
                out.extend(tokens[i : i + max_repeats * n])
                dropped += (reps - max_repeats) * n
                break
        if cut:
            i += cut
        else:
            out.append(tokens[i])
            i += 1
    return out, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--variants", default="sdm,ihm")
    parser.add_argument("--max-repeats", type=int, default=3)
    args = parser.parse_args()
    for variant in args.variants.split(","):
        src_dir = asr_run_dir(args.src, variant)
        for raw_path in sorted(src_dir.glob("*.raw.json")):
            meeting = raw_path.name.removesuffix(".raw.json")
            raw = read_json(raw_path)
            total = 0
            for chunk in raw["chunks"]:
                tokens, dropped = collapse(chunk["text"].split(), args.max_repeats)
                chunk["text"] = " ".join(tokens)
                total += dropped
            result = to_contract(raw)
            meta = read_json(src_dir / f"{meeting}.json")["meta"]
            result["meta"] = {**meta, "model": args.dst, "loop_collapse": {"max_repeats": args.max_repeats, "dropped_words": total}}
            write_json(asr_run_dir(args.dst, variant) / f"{meeting}.json", result)
            print(f"{args.dst}/{variant}/{meeting}: dropped {total} loop words")


if __name__ == "__main__":
    main()
