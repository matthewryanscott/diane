"""DER against the forced-alignment AMI references that NVIDIA's model card uses.

The card scores AMI with the nttcslab-sp/diar-forced-alignment RTTMs, which cut
within-segment silence out of the reference. The harness's pyannote
`only_words` references keep more of it (ES2004a: 923 s of speaker time vs
708 s). This script scores any diarizer run against the forced-alignment
references (collar 0 and +/-0.25 s, overlap scored, whole-file UEM as in
score.py) so a port can be checked against the published numbers.

The references are copied from https://github.com/nttcslab-sp/diar-forced-alignment
(AMI/test/<M>.rttm, commit 9527b7c) into data/ami/ref/rttm_fa/. Our 8 meetings
are half of the card's 16-meeting test set, so expect a small gap.

Usage: uv run python nemotron_fa_score.py --names nemotron3-mlx,nemotron3-fluidaudio,senko [--variants sdm,ihm]
"""

import argparse
import json
import warnings

from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

from common import DATA_DIR, MEETINGS, RESULTS_DIR, diar_run_dir, read_rttm, ref_uem_path

FA_REF_DIR = DATA_DIR / "ref" / "rttm_fa"


def _annotation(segments: list[tuple[str, float, float]], uri: str) -> Annotation:
    ann = Annotation(uri=uri)
    for i, (speaker, start, end) in enumerate(segments):
        ann[Segment(start, end), i] = speaker
    return ann


def score(name: str, variant: str) -> dict:
    der0 = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    der25 = DiarizationErrorRate(collar=0.5, skip_overlap=False)
    per = {}
    for m in MEETINGS:
        ref = _annotation(read_rttm(FA_REF_DIR / f"{m}.rttm"), m)
        hyp = _annotation(read_rttm(diar_run_dir(name, variant) / f"{m}.rttm"), m)
        fields = ref_uem_path(m).read_text().split()
        uem = Timeline([Segment(float(fields[2]), float(fields[3]))], uri=m)
        per[m] = {"der_c0": der0(ref, hyp, uem=uem), "der_c25": der25(ref, hyp, uem=uem)}
    tot = der0["total"]
    return {
        "per_meeting": per,
        "pooled": {
            "der_c0": abs(der0),
            "der_c25": abs(der25),
            "miss_c0": der0["missed detection"] / tot,
            "fa_c0": der0["false alarm"] / tot,
            "conf_c0": der0["confusion"] / tot,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--names", required=True)
    parser.add_argument("--variants", default="sdm,ihm")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    results = {}
    print("| run | variant | DER c0 % | DER c.25 % | miss % | FA % | conf % |")
    print("|---|---|---|---|---|---|---|")
    for name in args.names.split(","):
        for variant in args.variants.split(","):
            r = score(name, variant)
            results[f"{name}/{variant}"] = r
            p = {k: round(v * 100, 2) for k, v in r["pooled"].items()}
            print(f"| {name} | {variant} | {p['der_c0']} | {p['der_c25']} | {p['miss_c0']} | {p['fa_c0']} | {p['conf_c0']} |")
    out = RESULTS_DIR / f"diar_fa__{'+'.join(args.names.split(','))}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
